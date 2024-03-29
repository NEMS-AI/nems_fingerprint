"""
Package for simulating analyte mass absorption events on NEMS devices
"""
import numpy as np

from scipy import stats

from . import euler_bernoulli_beam as ebb
from comsol_mesh import *


MASS_DALTON_SQRT = 4.07497125e-14


class AbsorptionEvents:
    """Container for the frequency-shifts and masses of analyte absorption events
    
    Parameters
    ----------
    masses : (n_events,) ndarray
        masses of analytes for each absorption event
    freq_shifts : (n_events, n_modes) ndarray
        frequency shifts of analyte adsorptions
    """

    def __init__(self, masses, freq_shifts):
        # Check inputs
        masses = np.asanyarray(masses)
        freq_shifts = np.asanyarray(freq_shifts)
        
        n_events = len(masses)
        m, n_modes = freq_shifts.shape

        if n_events != m:
            raise ValueError(
                'length of masses must match length of first axis freq_shift '
                )
        
        # Store
        self.masses = masses
        self.freq_shifts = freq_shifts

    @property
    def n_modes(self):
        """Return the number of modes measured for analyte absorption events"""
        return self.freq_shifts.shape[1]
    
    @property
    def n_events(self):
        """Return the number of mass absorption events"""
        return len(self.masses)

    def __len__(self):
        """Return number of mass absorption events"""
        return len(self.masses)
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(n_events={self.n_events}, '
            f'n_modes={self.n_modes}, mean_mass={self.masses.mean()})'
        )
    

class SimulatedAbsorptionEvents(AbsorptionEvents):
    """Container for synthetically generated analyte absorption events
    
    This container extends the base class to store additional information such
    as the mode numbers and the positions of analyte absorption. Additional
    information can be stored by providing keywords arguments to the constructor.
    """
    def __init__(self, masses, freq_shifts, **kwargs):
        super().__init__(masses, freq_shifts)
        
        for (key, value) in kwargs.items():
            setattr(self, key, value)


class Distribution:
    """Class for generating random samples from a distribution
    
    This class wraps existing solutions to provide a consistent interface.

    Parameters
    ----------
    sample_generator : callable(size_tup=(n1, n2, ..., n_m))
        callable which returns array of random samples

    Examples
    --------
    >>> dist = Distribution.constant(1.0)
    >>> sample = dist.sample((3,))
    [1.0, 1.0, 1.0]
    """
    def __init__(self, sample_generator):
        self._sample_generator = sample_generator

    def sample(self, size):
        return self._sample_generator(size)
    
    @classmethod
    def constant(cls, value):
        f = lambda size_tup: np.full(size_tup, fill_value=value)
        return cls(f)
    
    @classmethod
    def uniform(cls, lower, upper):
        dist = stats.uniform(loc=lower, scale=upper-lower)
        return cls.from_scipy_dist(dist)

    @classmethod
    def normal(cls, mean, std):
        dist = stats.norm(loc=mean, scale=std)
        return cls.from_scipy_dist(dist)

    @classmethod
    def from_scipy_dist(cls, dist):
        f = lambda size_tup: dist.rvs(size=size_tup)
        return cls(f)


def frequency_shifts(mode_shapes, masses, positions):
    """Return frequency-shifts of analyte absorptions
    
    Parameters
    ----------
    mode_shapes : callable((n_events, dim_pts) array) -> (n_events, n_modes)
        function return vector of mode displacements at each position
    masses : (n_events,) ndarray
        masses of analytes
    positions : (n_events,) ndarray
        positions of analytes
    """
    return -0.5 * masses[:, np.newaxis] * mode_shapes(positions) ** 2


def simulate_absorption(n_events, mode_shapes, mass_dist, position_dist, noise_dist):
    """Return simulated mass absorption events
    
    Parameters
    ----------
    n_events : int
        number of events
    mode_shapes : callable((n_events, dim_pts) array) -> (n_events, n_modes)
        function return vector of mode displacements at each position
    mass_dist : Distribution
        distribution of analyte masses
    position_dist : Distribution
        distribution of analyte positions
    noise_dist : Distribution
        distribution of noise in frequency shift measurements

    Returns
    -------
    : SimulatedAbsorptionEvents
        simulated events
    """
    masses = mass_dist.sample(n_events)
    positions = position_dist.sample(n_events)
    freq_shifts = frequency_shifts(mode_shapes, masses, positions)

    noise = noise_dist.sample(freq_shifts.shape)

    events = SimulatedAbsorptionEvents(
        masses=masses,
        freq_shifts=freq_shifts + noise,
        positions=positions
    )
    return events


class Simulation:
    """Abstract simulation class"""
    def __init__(self, mode_shapes, mass_dist, position_dist, noise_dist):
        self.mode_shapes = mode_shapes
        self.mass_dist = mass_dist
        self.position_dist = position_dist
        self.noise_dist = noise_dist

    def sample(self, n_events):
        """Return sample of n_events from simulation"""
        return simulate_absorption(
            n_events, self.mode_shapes, self.mass_dist, self.position_dist, 
            self.noise_dist
        )


class EBBSimulation(Simulation):
    """Mass absorption simulation on 1-dimensional Euler-Bernoulli Beam
    
    Parameters
    ----------
    boundary_type : 'clamped-clamped' | 'clamped-free'
        type of boundary condition
    mode_indices : list[int]
        indices of modes to use in experiment from (1, 2, 3, ...)
    mass_dist : Distribution
        distribution of analyte masses
    position_dist : Distribution
        distribution of analyte positions
    noise_dist : Distribution
        distribution of noise in frequency shift measurements
    """

    def __init__(self, boundary_type, mode_indices, **kwargs):
        # Define mode_shapes function

        self.boundary_type = boundary_type
        self.mode_indices = mode_indices
        
        mode_shapes = lambda position: ebb.displacement(
                boundary_type=boundary_type,
                mode=mode_indices,
                x=position
            ).T
        super().__init__(mode_shapes, **kwargs)


class COMSOLSimulation(Simulation):
    """Mass absorption simulation on 1-dimensional Euler-Bernoulli Beam
    
    Mass is assumed to be in units of Dalton

    Parameters
    ----------
    mesh : comsol_mesh.Mesh
        mesh
    surface : comsol_mesh.Surface
        surface
    modes_field : comsol_mesh.Field
        eigenmodes
    mode_indices : list[int]
        indices of modes to use in experiment from (0, 1, 2, 3, ...)
    mass_dist : Distribution
        distribution of analyte masses
    position_dist : Distribution
        distribution of analyte positions
    noise_dist : Distribution
        distribution of noise in frequency shift measurements
    """
    def __init__(self, mesh, surface, modes_field, mode_idxs, mass_dist, noise_dist):
        self.mesh = mesh
        self.surface = surface
        self.modes_field = modes_field
        self.mode_idxs = mode_idxs

        self.mass_dist = mass_dist
        self.noise_dist = noise_dist

        # compute modal masses
        self.modal_masses = modes_field.L2_norm(axis=-1)
        
    def sample(self, n_events):
        _, values = self.surface.random_value_sample(
            self.modes_field, n_samples=n_events
        )
        freq_shifts = -0.5 * np.linalg.norm(values, axis=-1) ** 2 / self.modal_masses
        freq_shifts = freq_shifts[:, self.mode_idxs]


        masses = self.mass_dist.sample(n_events)
        noise = self.noise_dist.sample(freq_shifts.shape)

        return AbsorptionEvents(masses, freq_shifts + noise)
