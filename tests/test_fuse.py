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
