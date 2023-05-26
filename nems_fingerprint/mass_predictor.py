from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from sklearn.neighbors import KNeighborsRegressor

class MassPredictor(ABC):
    """
    Predict analyte mass for frequency shift data. Data for all predictors is first normalized
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
        self.masses = masses
        self.lp_norm_subset = lp_norm_subset
        self.freq_shifts = freq_shifts
        self.fit()
        
    @abstractmethod
    def fit():
        pass

    def __call__(self, freq_shifts):
        pass
    

class NNPredictor(MassPredictor):
    def fit(self):
        self._predictor_constants = self.masses / self.lp_norm_subset
        self._tree = spatial.KDTree(
            data=self.freq_shifts / self.lp_norm_subset[:, np.newaxis]
        )

    def __call__(self, freq_shifts):
        mp_norms = np.linalg.norm(freq_shifts, axis=-1)
        unit_freq_shifts = freq_shifts / mp_norms[:, np.newaxis]

        _, idxs = self._tree.query(unit_freq_shifts)

        if np.any(idxs == len(self.learning_events)):
            raise ValueError('Neighbour not found')

        mass_predictions = self._predictor_constants[idxs] * mp_norms
        return mass_predictions
    
class NNPredictor2(MassPredictor):
    def fit(self):
         self.model = KNeighborsRegressor(n_neighbors=1)
         self.model.fit(self.freq_shifts / self.lp_norm_subset[:, np.newaxis], self.masses / self.lp_norm_subset)

    def __call__(self, freq_shifts):
        mp_norms = np.linalg.norm(freq_shifts, axis=-1)
        unit_freq_shifts = freq_shifts / mp_norms[:, np.newaxis]

        predicted_norms = self.model.predict(unit_freq_shifts)
        mass_predictions = predicted_norms * mp_norms
