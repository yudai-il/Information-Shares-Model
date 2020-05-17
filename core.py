import numpy as np
from functools import reduce
import warnings
from statsmodels.tsa.vector_ar import vecm


norm = np.linalg.norm


class InfoShares:
    """
    Reference From: https://rdrr.io/rforge/ifrogs/src/R/pdshare.R
    """
    def __init__(self, data, k_ar_diff, maxLag=None,deterministic = "ci"):
        self.y_t = data
        self.endog_names = data.columns
        self.k_ar_diff = k_ar_diff

        self.deterministic = deterministic

        if isinstance(k_ar_diff,int) and k_ar_diff <= 0:
            raise Exception("k_ar_diff must greater than 0")

        elif isinstance(k_ar_diff,str):
            if not k_ar_diff in ['aic','bic','hqic','fpe']:
                raise Exception("k_ar_diff must in ['aic','bic','hqic','fpe'] ")
            else:
                if not isinstance(maxLag,int):
                    raise Exception("maxLag must be Integer")
                self.k_ar_diff = self.select_k_ar_diff(maxLag)

        self.neqs = data.shape[1]

        if not self.neqs == 2:
            raise Exception("only support 2 variable series")

    def select_k_ar_diff(self,maxLag):
        res = vecm.select_order(self.y_t, maxlags=maxLag, deterministic=self.deterministic)
        k_ar_diff = res.__getattribute__(self.k_ar_diff)

        k_ar_diff += 1 if k_ar_diff == 0 else k_ar_diff
        return k_ar_diff

    def fit(self):

        alpha, beta, gamma, omega, corrcoefs = self.vecm()

        def _orth(vector):
            v = np.array(vector).flatten()
            return np.matrix([-1 * v[1], v[0]]).T

        orth_a = _orth(alpha)
        orth_b = _orth(beta)
        g = reduce(lambda x, y: np.add(x, y), gamma)
        pi = orth_a.T.dot(np.eye(len(alpha)) - g).dot(orth_b) ** (-1)
        psi = orth_b.dot(pi).dot(orth_a.T)

        if not np.allclose(np.linalg.det(psi),0):
            warnings.warn("Financial might not convergence")

        s1 = ishares(psi[0], omega)
        s2 = mishares(corrcoefs, psi[0], omega)
        cs = compshares(orth_a)

        results = {"infoShares": s1, "MInfoShares": s2,"compShares":cs,
                   "k":self.k_ar_diff,"a": alpha, "b": beta, "g": gamma,
                   "omega": omega, "corrcoefs": corrcoefs}

        return ResultsWarper(results)

    def get_clean_results(self, res):
        alpha = res.alpha
        beta = res.beta
        gamma = res.gamma

        u = res.resid
        gamma = np.array(np.hsplit(gamma, self.k_ar_diff))

        omega = np.cov(u, rowvar=False)
        corrcoefs = np.corrcoef(u, rowvar=False)

        return alpha, beta, gamma, omega, corrcoefs

    def vecm(self):
        k_ar_diff = self.k_ar_diff
        from statsmodels.tsa.vector_ar.vecm import VECM
        coint_rank = self.neqs - 1
        model = VECM(self.y_t, k_ar_diff=k_ar_diff, deterministic=self.deterministic, coint_rank=coint_rank)
        res = model.fit()
        return self.get_clean_results(res)


class ResultsWarper:
    def __init__(self, kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)




def compshares(orth_a):
    abs_ = abs(orth_a)
    return abs_/abs_.sum()


def ishares(psi, omega):
    total_variance = psi.dot(omega).dot(psi.T)

    f = np.linalg.cholesky(omega)
    proportion = np.power(psi.dot(f), 2) / total_variance

    return proportion


def mishares(corr, psi, omega):
    from scipy.linalg import eigh

    _lambda, evec = eigh(corr)
    _lambda = np.diag(_lambda)

    g = evec.transpose()
    v = np.diag(np.sqrt(np.diag(omega)))

    matrix_f_asterisk = np.dot(g.dot(np.power(np.linalg.inv(_lambda), .5)), g.transpose()).dot(np.linalg.inv(v))

    # matrix_f_asterisk = np.dot(g.dot(np.linalg.inv(_lambda**0.5)),g.transpose()).dot(np.linalg.inv(v))

    matrix_f_asterisk = np.linalg.inv(matrix_f_asterisk)

    phi_asterisk = psi.dot(matrix_f_asterisk)

    total_variance = psi.dot(omega).dot(psi.T)

    proportion_mis = np.power(phi_asterisk, 2) / total_variance

    return proportion_mis



