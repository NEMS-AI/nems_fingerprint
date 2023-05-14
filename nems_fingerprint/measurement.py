"""
Fingerprinting methods for predicting mass from analyte frequency shifts

Author: Alex Nunn
Date: 28/10/22
"""

import numpy as np

from scipy import spatial
from scipy import special
from synthetic import Sample


class MassPredictNN:
    """Predict analyte mass from frequency shift using nearest neighbour method

    To ensure the stability of the method, frequency shift samples with norm
    smaller than reject_tol are discarded. The default of reject_tol if no
    tolerance is provided is to reject vectors with norms less than 5% of the
    median norm.

    Parameters
    ----------
    freq_shift : ndarray(n, n_modes)
        array of n frequency shifts with n_modes number of modes to use for
        learning database
    mass : float | ndarray(n,)
        masses of analytes in the learning database
    reject_tol : float
        reject vectors with norm smaller than reject_tol from the dataset
        (default: reject vectors smaller than 5% of median norm)


    Attributes
    ----------
    reject_tol : float
        rejection tolerance used to filter frequency shifts
    learning_set : synthetic.Sample
        container of learning set used

    Methods
    -------
    __call__(freq_shift)
        return mass prediction for frequency shift

    Examples
    --------
    Creating a predictor
    >>> learning_config = synthetic.BeamExperiment(
            boundary_type='clamped-free',
            modes=6
        )
    >>> sample = learning_config.sample(n_samples=1000)
    >>> measurement_config = learning_config.identical_except_for(mass=1)
    >>> measurement_sample = measurement_config.sample(n_samples=10)
    >>> mass_predictor = fingerprint.MassPredictNN(
            freq_shift=sample.freq_shift,
            mass=sample.mass
        )
    >>> mass_predictor(measurement_sample.freq_shift)
    array([-9.54189236e-05,  1.29060331e-04, -9.67982241e-04,  2.33821125e-03,
        5.72173650e-04, -3.63194538e-05,  1.06661675e-03,  1.36725395e-03,
       -6.71446052e-04,  5.84691820e-04])
    """

    def __init__(self, freq_shift, mass, reject_tol=None):
        # Compute norms
        norms = np.linalg.norm(freq_shift, axis=-1)

        # Enforce reject_tol default
        if reject_tol is None:
            reject_tol = np.median(norms) * 0.05
        self._reject_tol = reject_tol

        # Filter dataset
        selection = norms > reject_tol

        # Store sample learning set
        self._learning_set = Sample(
            freq_shift=freq_shift[selection],
            mass=mass[selection]
        )

        # Reject samples with norms less than reject_tol
        self._predictor_mass = mass[selection] / norms[selection]
        self._kdtree = spatial.KDTree(
            data=freq_shift[selection] / norms[selection][:, np.newaxis]
        )

    def __call__(self, freq_shift):
        norm = np.linalg.norm(freq_shift, axis=-1)
        _, idx = self._kdtree.query(freq_shift / norm[:, np.newaxis])

        if np.any(idx == len(self._learning_set.freq_shift)):
            raise ValueError('Neighbour not found')

        mass_estimate = self._predictor_mass[idx] * norm
        return mass_estimate

    @property
    def reject_tol(self):
        return self._reject_tol

    @property
    def learning_set(self):
        return self._learning_set

    def __repr__(self):
        return f'{self.__class__.__name__}(learning_set={self.learning_set})'


class MassPredictRegressionLC:
    pass


class MassPredictRegressionLL:
    pass


def kernel_langevin(d, kappa, theta):
    r"""Return value of Langevin kernel

    For a sphere d-1 embedded in d-dimensional Euclidean space the value of the
    Langevin kernel is given by

        K_\kappa(\cos \theta) = \kappa^{d/2 - 1} \exp{\kappa \cos\theta}
    """
    return kappa
