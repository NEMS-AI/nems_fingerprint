from .mass_predictor import MassPredictor
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from scipy.stats import ttest_1samp

class thetaSensibility(MassPredictor):
    """
    Classify measurement sensibility given frequency shift data. Data for all predictors is first normalized
    onto the unit sphere and frequency shift samples with norm
    smaller than reject_tol are discarded.

    Parameters
    ----------
    learning_events : AbsorptionEvents
        container of masses and frequency shifts to use in learning phase
    reject_tol : float
        frequency shifts with a norm smaller than reject_tol are excluded from
        the learning phase. If no tolerance is provided the tolerance is set to
        the 5% quantile of learning phase norms.
    """
    def fit(self):
        unit_freq_shifts=self.freq_shifts / self.lp_norm_subset[:, np.newaxis]
        self.average_point = np.mean(unit_freq_shifts, axis=0)
        dot_products = np.dot(unit_freq_shifts, self.average_point)
        self.angles = np.arccos(dot_products)
        self.alpha = 0.05

        print("model_trained")

    def __call__(self, freq_shifts):
        mp_norms = np.linalg.norm(freq_shifts, axis=-1)
        unit_freq_shifts = freq_shifts / mp_norms[:, np.newaxis]

        m_dist = np.zeros(freq_shifts.shape[0])
        for i in range(freq_shifts.shape[0]):
            new_dot_products = np.dot(unit_freq_shifts, self.average_point)
            self.new_angles = np.arccos(new_dot_products)
            _, p_val = ttest_1samp(self.angles, self.new_angles[i])
            m_dist[i] = p_val

        sensible_meas = m_dist <= self.alpha
        return sensible_meas


class chi2Sensiblity(MassPredictor):
    """
    Classify measurement sensibility given frequency shift data. Data for all predictors is first normalized
    onto the unit sphere and frequency shift samples with norm
    smaller than reject_tol are discarded.

    Parameters
    ----------
    learning_events : AbsorptionEvents
        container of masses and frequency shifts to use in learning phase
    reject_tol : float
        frequency shifts with a norm smaller than reject_tol are excluded from
        the learning phase. If no tolerance is provided the tolerance is set to
        the 5% quantile of learning phase norms.
    """
    def fit(self):
        data=self.freq_shifts / self.lp_norm_subset[:, np.newaxis]
        self.mu = np.mean(data, axis=0)
        self.cov = np.cov(data, rowvar=False)
        self.inv_cov = np.linalg.inv(self.cov)
        self.alpha = 0.05

        print("model_trained")

    def __call__(self, freq_shifts):
        mp_norms = np.linalg.norm(freq_shifts, axis=-1)
        unit_freq_shifts = freq_shifts / mp_norms[:, np.newaxis]

        m_dist = np.zeros(freq_shifts.shape[0])
        for i in range(freq_shifts.shape[0]):
            m_dist[i] = mahalanobis(freq_shifts[i], self.mu, self.inv_cov)
        sensible_meas = m_dist <= np.sqrt(chi2.ppf(1 - self.alpha, len(self.mu)))
        return sensible_meas