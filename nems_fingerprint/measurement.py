"""
Fingerprinting methods for predicting mass from analyte frequency shifts

Author: Alex Nunn
Date: 28/10/22
"""

import numpy as np
from scipy import spatial


class MassPredictNN:
    """Predict analyte mass from frequency shift using nearest neighbour method

    Parameters
    ----------
    learning_events : AbsorptionEvents
        container of masses and frequency shifts to use in learning phase
    reject_tol : float
        frequency shifts with a norm smaller than reject_tol are excluded from
        the learning phase. If no tolerance is provided the tolerance is set to
        the 5% quantile of learning phase norms.

    Examples
    --------
    Create a predictor

    >>> ebb_sim = EBBSimulation(
        boundary_type='clamped-free',
        modes=[1, 2, 3],
        mass_dist=Distribution.constant(1.0),
        position_dist=Distribution.uniform(0.0, 1.0),
        noise_dist=Distribution.normal(0.0, 0.01)
    )
    >>> learning_events = ebb_sim.sample(n_events=1000)
    >>> mass_predictor = MassPredictNN(learning_events)
    >>> measurement_phase_shifts = ebb_sim.sample(n_events=20).freq_shifts
    >>> masses = mass_predictor(measurement_phase_shifts)
    """

    def __init__(self, learning_events, reject_tol=None):

        # Preprocess Data
        lp_norms = np.linalg.norm(learning_events.freq_shifts, axis=-1)

        # Default rejection tolerance
        if reject_tol is None:
            reject_tol = np.quantile(lp_norms, 0.05)
        
        # Select learning events
        selection = lp_norms > reject_tol
        freq_shifts = learning_events.freq_shifts[selection]
        masses = learning_events.masses[selection]
        lp_norm_subset = lp_norms[selection]

        # Store
        self.reject_tol = reject_tol
        self.learning_events = learning_events
        self.selection = selection

        self._predictor_constants = masses / lp_norm_subset
        self._tree = spatial.KDTree(
            data=freq_shifts / lp_norm_subset[:, np.newaxis]
        )
    
    def __call__(self, freq_shifts):
        mp_norms = np.linalg.norm(freq_shifts, axis=-1)
        unit_freq_shifts = freq_shifts / mp_norms[:, np.newaxis]

        _, idxs = self._tree.query(unit_freq_shifts)

        if np.any(idxs == len(self.learning_events)):
            raise ValueError('Neighbour not found')

        mass_predictions = self._predictor_constants[idxs] * mp_norms
        return mass_predictions
    