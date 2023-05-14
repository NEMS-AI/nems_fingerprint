"""
Synthetic data generation

TODO: Implement better reprs for Sample and BeamExperiment

Author: Alex Nunn
Date: 28/10/22

"""
import numpy as np
from . import euler_bernoulli_beam as ebb

from typing import Callable
from dataclasses import dataclass
from scipy import stats

# TODO: Add better repr for sample class
# TODO: Create class decorator that gives pretty __repr_html__
@dataclass
class Sample:
    """Container for synthetic experiment data"""
    modes: list = None
    mass: list = None
    position: list = None
    noise: list = None
    freq_shift: list = None

    def __len__(self):
        return len(self.mass)


@dataclass
class BeamExperiment:
    """Configuration of artifical Euler Bernoulli experiment

    Parameters
    ----------
    boundary_type: str
        type of boundary configuration of Euler-Bernoulli beam

    modes : int | list[int]
        modes to measure in artificial experiment. If an integer uses modes
        1, 2, ..., modes otherwise uses subset of modes defined by list

    mass_distribution : float | Callable | scipy.stats.rvs (optional)
        mass distribution of analytes to sample from. If a float the every mass
        is identical. If callable then
            mass_distribution(n)
        should return n samples from the mass distribution. If scipy.stats.rvs
        then the .rvs() method is called. Default behaviour is an identical
        distribution of 1.

    noise_distribution : float | Callable | scipy.stats.rvs (optional)
        distribution of noise for frequency measurements. If float then constant
        noise is added to frequency measurements. If Callable then
            noise_distribution(n)
        is called for (n, modes) noise samples. If scipy.stats.rvs then .rvs()
        method is called. Default behaviour is 0, for no experimental noise.

    position_distriubtion : Callable | scipy.stats.rsv (optional)
        distribution of points along the beam. Calling behaviour for Callable
        and scipy.stats.rsv is the same as above. The default is a uniform
        distribution.
    """
    boundary_type: str
    modes: int
    mass_distribution: float = 1
    position_distribution: Callable = stats.uniform()
    noise_distribution: Callable = 0

    def __post_init__(self):
        # Assert defaults behaviour
        if type(self.modes) is int:
            self.modes = list(range(1, self.modes + 1))

    def theoretical_freq_shift(self, mass, position):
        """Returns the frequency shift predicted by theory

        Parameters
        ----------
        mass : ndarray(n,)
            sample of masses
        position : ndarray(n,)
            sample of positions

        Returns
        -------
        freq_shifts : ndarray(n, modes)
            frequency shifts
        """
        if np.isscalar(mass):
            mass = np.array([mass])

        return -0.5 * mass[:, np.newaxis] * ebb.displacement(
                boundary_type=self.boundary_type,
                mode=self.modes,
                x=position
            ).T ** 2

    def sample(self, n_samples):
        """Return n samples from experiment

        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        Sample
        """
        mass_sample = self._sample_dist(self.mass_distribution, n_samples)
        position_sample = self._sample_dist(self.position_distribution, n_samples)
        noise_sample = self._sample_dist(
            self.noise_distribution, n_samples, len(self.modes)
        )

        freq_shift = (
            self.theoretical_freq_shift(mass_sample, position_sample)
            + noise_sample
        )

        sample = Sample(
            modes=self.modes,
            mass=mass_sample,
            position=position_sample,
            noise=noise_sample,
            freq_shift=freq_shift
        )
        return sample

    def _sample_dist(self, dist, *args):
        if np.isscalar(dist):
            return np.full(shape=args, fill_value=dist)
        try:
            sample = dist(*args)
            return sample
        except TypeError:
            sample = dist.rvs(size=args)
            return sample

    def identical_except_for(self, **kwargs):
        keys = (
            'boundary_type', 'modes', 'mass_distribution',
            'position_distribution', 'noise_distribution'
        )
        props = {k: getattr(self, k) for k in keys}
        props.update(kwargs)
        return self.__class__(**props)

    def __repr__(self):
        return f'''{self.__class__.__name__}(
    modes={self.modes},
    mass_distribution={self.mass_distribution},
    noise_distribution={self.noise_distribution},
    position_distribution={self.position_distribution}
)'''
