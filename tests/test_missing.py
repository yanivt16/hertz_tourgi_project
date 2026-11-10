import numpy as np
import pytest

from fuse import FUSE, mom_estimate, moment_tensors
from fuse import missing as mi
from fuse.tci import minimize_tci


def simulate(n, psi, eta, prevalence, rng, observe=1.0, observe_pos=None):
    y = rng.random(n) < prevalence
    u = rng.random((n, len(psi)))
    V = np.where(np.where(y[:, None], u < psi, u >= eta), 1.0, -1.0)
    p = np.where(y[:, None], observe if observe_pos is None else observe_pos, observe)
    V[rng.random(V.shape) >= p] = np.nan
    return V, y


def scores_V(rng, n=400, m=8):
    y = rng.random(n) < 0.4
    return y[:, None] * rng.uniform(0.3, 1.0, m) + rng.normal(0, 1, (n, m)), y


def test_masked_moments_equal_full_moments_without_missing():
    rng = np.random.default_rng(0)
    V, _ = scores_V(rng)
    full, mm = moment_tensors(V), mi.masked_moments(V)
    np.testing.assert_allclose(mm.mu, full.mu)
    np.testing.assert_allclose(mm.sigma, full.sigma)
    np.testing.assert_allclose(mm.T, full.T)


def test_masked_tci_search_equals_full_search_without_missing():
    rng = np.random.default_rng(1)
    V, _ = scores_V(rng)
    a, b = minimize_tci(V), mi.minimize_tci_masked(V)
    np.testing.assert_array_equal(a.tau, b.tau)
    assert a.statistic == pytest.approx(b.statistic)


@pytest.mark.parametrize("predictor", ["logistic", "posterior"])
def test_skip_mode_equals_default_fuse_without_missing(predictor):
    rng = np.random.default_rng(2)
    V, _ = scores_V(rng)
    a = FUSE(predictor=predictor).fit_scores(V)
    b = FUSE(predictor=predictor, missing="skip").fit_scores(V)
    np.testing.assert_allclose(a.scores, b.scores, atol=1e-10)


def test_masked_mom_recovers_parameters_with_random_missingness():
    rng = np.random.default_rng(3)
    psi = np.array([0.9, 0.8, 0.75, 0.7, 0.85, 0.65])
    eta = np.array([0.7, 0.85, 0.8, 0.6, 0.75, 0.9])
    V, _ = simulate(300_000, psi, eta, 0.3, rng, observe=0.4)
    est = mi.masked_mom_estimate(V)
    assert est.prevalence == pytest.approx(0.3, abs=0.03)
    np.testing.assert_allclose(est.psi, psi, atol=0.03)
    np.testing.assert_allclose(est.eta, eta, atol=0.03)
    # zero-imputation on the same data is biased
    biased = mom_estimate(np.nan_to_num(V))
    assert np.abs(biased.psi - psi).max() > 0.1


def test_missing_entries_do_not_vote_in_the_posterior():
    est = mi.MoMEstimate(psi=np.full(3, 0.8), eta=np.full(3, 0.8), b=0.0, u=np.zeros(3), mu=np.zeros(3))
    V = np.array([[1.0, 1.0, np.nan], [1.0, 1.0, 1.0]])
    p = mi.product_posterior(V, est)
    assert p[0] == pytest.approx(0.8**2 / (0.8**2 + 0.2**2))
    assert p[1] > p[0]


def test_fuse_skip_beats_zero_imputation_on_sparse_data():
    rng = np.random.default_rng(4)
    m = 20
    psi, eta = rng.uniform(0.6, 0.85, m), rng.uniform(0.6, 0.85, m)
    V, y = simulate(3000, psi, eta, 0.3, rng, observe=0.25)
    skip = FUSE(missing="skip", predictor="posterior").fit_scores(V, rescale=False).scores
    zero = FUSE(predictor="posterior").fit_scores(np.nan_to_num(V), rescale=False).scores
    assert ((skip > 0.5) == y).mean() > ((zero > 0.5) == y).mean() + 0.02


def test_ordinal_dawid_skene_recovers_labels_from_scores():
    from fuse.ordinal import dawid_skene_ordinal, quantile_bins

    rng = np.random.default_rng(5)
    y = rng.random(2000) < 0.3
    V = y[:, None] * rng.uniform(1.0, 2.0, 8) + rng.normal(0, 1, (2000, 8))
    assert set(np.unique(quantile_bins(V, 5))) == {0, 1, 2, 3, 4}
    q = dawid_skene_ordinal(V, 5)
    assert ((q > 0.5) == y).mean() > 0.9
    assert q.mean() == pytest.approx(0.3, abs=0.05)
