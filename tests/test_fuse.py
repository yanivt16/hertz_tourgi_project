import numpy as np
import pytest

from fuse import FUSE, fuse_batched, mom_estimate, moment_tensors, selection_accuracy, tci_statistic
from fuse.baselines import naive_ensemble_scores
from fuse.posterior import averaged_triplet_posterior, triplet_posterior
from fuse.moments import MoMEstimate
from fuse.tci import minimize_tci


def simulate_binary(n, psi, eta, prevalence, rng):
    y = rng.random(n) < prevalence
    u = rng.random((n, len(psi)))
    pos = np.where(y[:, None], u < psi[None, :], u >= eta[None, :])
    return np.where(pos, 1.0, -1.0), y


def test_mom_recovers_parameters_under_conditional_independence():
    rng = np.random.default_rng(0)
    psi = np.array([0.9, 0.8, 0.75, 0.7, 0.85, 0.65])
    eta = np.array([0.7, 0.85, 0.8, 0.6, 0.75, 0.9])
    V, _ = simulate_binary(200_000, psi, eta, prevalence=0.3, rng=rng)
    est = mom_estimate(V)
    assert est.b == pytest.approx(0.3 - 0.7, abs=0.03)
    np.testing.assert_allclose(est.psi, psi, atol=0.03)
    np.testing.assert_allclose(est.eta, eta, atol=0.03)


def test_mom_resolves_sign_with_a_minority_of_adversarial_verifiers():
    rng = np.random.default_rng(1)
    psi = np.array([0.85, 0.8, 0.8, 0.75, 0.2])
    eta = np.array([0.8, 0.75, 0.85, 0.8, 0.25])
    V, _ = simulate_binary(100_000, psi, eta, prevalence=0.5, rng=rng)
    est = mom_estimate(V)
    assert est.balanced_accuracy[-1] < 0.5
    assert np.all(est.balanced_accuracy[:-1] > 0.5)


def test_tci_statistic_separates_independent_from_dependent_verifiers():
    rng = np.random.default_rng(2)
    psi = np.full(6, 0.8)
    eta = np.full(6, 0.75)
    V, y = simulate_binary(100_000, psi, eta, prevalence=0.4, rng=rng)
    m_ind = moment_tensors(V)
    independent = tci_statistic(m_ind.sigma, m_ind.T)

    # Make verifiers 1-3 near-copies of verifier 0: strong conditional dependence.
    Vd = V.copy()
    for j in (1, 2, 3):
        flip = rng.random(len(V)) < 0.05
        Vd[:, j] = np.where(flip, -V[:, 0], V[:, 0])
    m_dep = moment_tensors(Vd)
    dependent = tci_statistic(m_dep.sigma, m_dep.T)
    assert independent < 1e-3
    assert dependent > 10 * independent


def test_moment_tensor_chunking_matches_einsum():
    rng = np.random.default_rng(3)
    V = rng.normal(size=(9000, 5))
    mt = moment_tensors(V)
    X = V - V.mean(0)
    np.testing.assert_allclose(mt.T, np.einsum("ni,nj,nk->ijk", X, X, X) / len(V), atol=1e-10)


def test_triplet_posterior_matches_exact_bayes():
    psi = np.array([0.9, 0.7, 0.6])
    eta = np.array([0.8, 0.65, 0.9])
    b = 0.2
    est = MoMEstimate(psi=psi, eta=eta, b=b, u=np.zeros(3), mu=np.zeros(3))
    V = np.array([[1, 1, -1], [-1, 1, -1], [1, -1, 1]], dtype=float)
    p = triplet_posterior(V, est, (0, 1, 2))
    prior = (1 + b) / 2
    for row, got in zip(V, p):
        like_pos = np.prod(np.where(row > 0, psi, 1 - psi))
        like_neg = np.prod(np.where(row > 0, 1 - eta, eta))
        expected = prior * like_pos / (prior * like_pos + (1 - prior) * like_neg)
        assert got == pytest.approx(expected)
    np.testing.assert_allclose(averaged_triplet_posterior(V, est), p)


def test_threshold_search_never_increases_the_statistic():
    rng = np.random.default_rng(4)
    n = 2000
    y = rng.random(n) < 0.3
    latent = rng.normal(size=n)
    V = np.column_stack(
        [np.where(y, rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n)) + (0.8 * latent if j < 3 else 0.0) for j in range(6)]
    )
    start = np.median(V, axis=0)
    m0 = moment_tensors(np.where(V > start, 1.0, -1.0))
    res = minimize_tci(V, n_sweeps=3)
    assert res.statistic <= tci_statistic(m0.sigma, m0.T) + 1e-12


def test_fuse_weights_verifiers_by_estimated_quality():
    """Two strong, five mediocre and two adversarial verifiers: FUSE should at least match
    the naive average, which the adversarial pair drags down."""
    rng = np.random.default_rng(8)
    means = [2.5, 2.5, 0.6, 0.6, 0.6, 0.6, 0.6, -0.8, -0.8]
    fuse_acc, naive_acc = [], []
    for _ in range(30):
        n = 100
        y = rng.random(n) < 0.3
        V = np.column_stack([np.where(y, rng.normal(mu, 1.0, n), rng.normal(0.0, 1.0, n)) for mu in means])
        fuse_acc.append(selection_accuracy(FUSE().fit_scores(V).scores, y))
        naive_acc.append(selection_accuracy(naive_ensemble_scores(V), y))
    assert np.mean(fuse_acc) >= np.mean(naive_acc)
    assert np.mean(fuse_acc) > 0.95


def test_fuse_falls_back_when_verifiers_are_constant():
    V = np.ones((10, 4))
    V[:, 0] = np.arange(10)
    res = FUSE().fit_scores(V)
    assert res.fallback is not None
    assert res.select() == 9


def test_batched_mode_returns_scores_per_query():
    rng = np.random.default_rng(6)
    mats = [rng.normal(size=(20, 5)) for _ in range(4)]
    scores, _ = fuse_batched(mats)
    assert [len(s) for s in scores] == [20] * 4


def test_selection_accuracy_splits_ties():
    assert selection_accuracy(np.array([1, 1, 0]), np.array([1, 0, 1])) == pytest.approx(0.5)


def test_standardized_tci_statistic_is_scale_free_and_zero_under_tci():
    rng = np.random.default_rng(9)
    V, _ = simulate_binary(100_000, np.full(6, 0.8), np.full(6, 0.75), prevalence=0.4, rng=rng)
    m = moment_tensors(V)
    assert tci_statistic(m.sigma, m.T, standardize=True) < 1e-2
    scaled = moment_tensors(V * np.array([1, 2, 0.5, 3, 1, 1]))
    assert tci_statistic(scaled.sigma, scaled.T, clip=1e-6, standardize=True) == pytest.approx(
        tci_statistic(m.sigma, m.T, clip=1e-6, standardize=True), rel=1e-6
    )


def test_clustering_merges_a_correlated_block_and_keeps_independent_verifiers_apart():
    from fuse.clustering import cluster_verifiers

    rng = np.random.default_rng(10)
    n = 5000
    y = rng.random(n) < 0.4
    nuisance = rng.normal(size=n)
    good = [np.where(y, 1.5, 0.0) + rng.normal(size=n) for _ in range(3)]
    junk = [nuisance + 0.3 * rng.normal(size=n) for _ in range(5)]
    junk[2] = -junk[2]  # a reversed copy still belongs to the block
    clusters = cluster_verifiers(np.column_stack(good + junk), threshold=0.5)
    sets = sorted(sorted(g.tolist()) for g in clusters.groups)
    assert sets == [[0], [1], [2], [3, 4, 5, 6, 7]]
    assert clusters.signs[5] == -clusters.signs[3]


def test_clustering_rescues_fuse_from_a_correlated_junk_block():
    """Nine near-random verifiers driven by one shared factor outvote three good,
    independent ones in the rank-one moment fit; collapsing the block restores them."""
    rng = np.random.default_rng(11)
    plain, clustered = [], []
    for _ in range(20):
        n = 100
        y = rng.random(n) < 0.3
        nuisance = rng.normal(size=n)
        good = [np.where(y, 1.5, 0.0) + rng.normal(size=n) for _ in range(3)]
        junk = [nuisance + 0.15 * y + 0.5 * rng.normal(size=n) for _ in range(9)]
        V = np.column_stack(good + junk)
        plain.append(selection_accuracy(FUSE().fit_scores(V).scores, y))
        clustered.append(selection_accuracy(FUSE(cluster_threshold=0.5).fit_scores(V).scores, y))
    assert np.mean(clustered) > np.mean(plain) + 0.1


def test_majority_vote_verifier_appends_answer_agreement():
    from fuse.data import Benchmark, Query
    from fuse.experiment import with_majority_vote

    q = Query("q", np.zeros((4, 2)), np.array([1, 1, 0, 0], bool), answers=["a", "a", "b", None])
    out = with_majority_vote([q.scores], Benchmark("t", ["v1", "v2"], [q]))
    np.testing.assert_array_equal(out[0][:, -1], [2, 2, 1, -1])


def _scores_with_block(n, rng, nuisance):
    """Three conditionally independent verifiers plus a correlated block of four.

    nuisance=True: the block shares a factor unrelated to correctness and barely sees the label.
    nuisance=False: the block is correlated only because every member is accurate.
    """
    y = rng.random(n) < 0.5
    s = np.where(y, 1.0, -1.0)
    good = [0.8 * s + rng.normal(size=n) for _ in range(3)]
    z = rng.normal(size=n)
    if nuisance:
        block = [0.1 * s + 1.5 * z + 0.5 * rng.normal(size=n) for _ in range(4)]
    else:
        block = [2.0 * s + rng.normal(size=n) for _ in range(4)]
    return np.column_stack(good + block), y


@pytest.mark.parametrize("nuisance", [True, False])
def test_select_variant_collapses_nuisance_blocks_but_not_informative_ones(nuisance):
    from fuse.selection import label_share, select_variant

    rng = np.random.default_rng(7)
    mats = [_scores_with_block(200, rng, nuisance)[0] for _ in range(20)]
    sel = select_variant(mats)
    if nuisance:
        assert sel.cluster_threshold is not None
        assert sel.variant.startswith("cluster@")
        assert label_share(np.vstack(mats), 0.5) < 0.2
    else:
        assert sel.cluster_threshold is None
        assert sel.variant == "base"
        assert label_share(np.vstack(mats), 0.5) > 0.8


def test_select_variant_adds_majority_vote_only_when_it_fits_the_label_model():
    from fuse.selection import select_variant

    rng = np.random.default_rng(8)
    mats, informative, noise = [], [], []
    for _ in range(20):
        V, y = _scores_with_block(200, rng, nuisance=False)
        mats.append(V)
        informative.append(np.where(y, 1.0, -1.0) + 0.5 * rng.normal(size=len(y)))
        noise.append(rng.normal(size=len(y)))
    assert select_variant(mats, informative).majority_vote
    assert not select_variant(mats, noise).majority_vote
    assert select_variant(mats, informative).variant == "mv"


def test_rank_one_fit_ignores_masked_entries():
    from fuse.moments import rank_one_offdiagonal

    u = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    S = np.outer(u, u)
    S[0, 1] = S[1, 0] = 0.99  # corrupted entry, excluded from the fit
    observed = ~np.eye(5, dtype=bool)
    observed[0, 1] = observed[1, 0] = False
    np.testing.assert_allclose(np.abs(rank_one_offdiagonal(S, observed=observed)), u, atol=1e-6)
