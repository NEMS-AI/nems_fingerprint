# NEMS Fingerprint
A Python package for mass spectrometry via Nano-electromechanical Systems (NEMS) using the **fingerprint method**.

## Description
The **fingerprint method** for NEMS mass spectrometry enables the mass a nanoscale analyte to be determined from the frequency shifts induced in each oscillation mode as the analyte absorbs to the surface of the NEMS device. By calibrating with analytes of known mass in the _learning phase_ the fingerprint method can develop a _fingerprint database_ which characterizes the response of a NEMS device. The masses of arbitrary analytes can then be determined by matching the measured frequency shifts to the __fingerprint database_ in the _measurement phase_.

The **fingerprint method** provides an alternative to previous NEMS mass spectrometry methods which rely on simplified analytic models or finite element simulations of the oscillatory modes of the NEMS device. By removing the requirement for an underlying model of the oscillatory modes and calibrating on the device directly, the fingerprint method extends NEMS mass spectrometry to devices with complex uncharacterized modes.

## Usage
The masses of analytes can be predicted using the known masses and frequency shifts of the learning phase.

```python
from nems_fingerprint import *

# Learning phase -- artificial example
masses = [1.0, 1.0, 1.1, 0.9]  # masses of calibration analytes
freq_shifts = [
    [1.0, 0.95, 0.82],  # frequency shifts due to analyte 1
    [1.1, 0.85, 0.81],  # ... analyte 2
    [1.0, 0.79, 0.83],  # ... etc
    [1.1, 0.93, 0.84],
]

learning_events = AbsorptionEvents(masses, freq_shifts)
mass_predictor = MassPredictNN(learning_events)

# Measurement phase
freq_shifts_measurement_phase = [
    [1.0, 0.75, 0.80],  # frequency shifts due to analyte 1 of unknown mass
    [1.0, 0.93, 0.84],  # frequency shifts due to analyte 1 of unknown mass
]

recovered_masses = mass_predictor(freq_shifts_measurement_phase)
```

Absorptions events on NEMS devices can be simulated using Euler-Bernoulli beam theory using the class `EBBSimulation`.

```python
from nems_fingerprint import *

# Learning phase
#   simulate absorptions on slender Euler-Bernoulli beam
ebb_sim = EBBSimulation(
    boundary_type='clamped-free',               # define boundary conditions
    modes=[1, 2, 3],                            # defines modes simulated
    mass_dist=Distribution.constant(1.0),       # distriubtion of calibration analytes
    position_dist=Distribution.uniform(0.0, 1.0),  # position distribution
    noise_dist=Distribution.normal(0.0, 0.01)      # noise distribution
)

learning_events = ebb_sim.sample(n_events=1000)
mass_predictor =  MassPredictNN(learning_events)

# Measurement phase
events = ebb_sim.sample(n_events=20)
masses = mass_predictor(events.freq_shifts)
```

To simulate frequency adsorption events on NEMS devices with bulk 3 dimensional resonant modes the eigenmodes can be computed using the industry standard finite-element solver COMSOL.


```python


def surface_sample_generator(n_samples):
    x0 = 0.3
    y1, y2 = 0.2, 1.2
    z1, z2 = -0.6, 0.6
    
    samples = np.zeros((n_samples, 3))
    samples[:, 0] = x0
    samples[:, 1] = np.random.uniform(y1, y2, n_samples)
    samples[:, 2] = np.random.uniform(z1, z2, n_samples)
    return samples

position_dist = Distribution(surface_sample_generator)

comsol_sim = COMSOLSimulation(
    'COMSOL_device_eigenmodes.csv',             # path to COMSOL eigenmodes export
    mode_indices=[0, 1, 2],                     # 1st, 2nd, 3rd mode (zero indexing)
    mass_dist=Distribution.constant(1000e3),    # constant mass 1000kDa
    position_dist=position_dist,
    noise_dist=Distribution.constant(0.0)       # no noise
)

learning_events = comsol_sim.sample(n_events=10000)
mass_predictor = MassPredictNN(learning_events)

measurement_phase_shifts = comsol_sim.sample(n_events=2000).freq_shifts
masses_rel = mass_predictor(measurement_phase_shifts) / M0
masses_rel_err = masses_rel - 1
```

For further usage examples see notebooks in `examples/` directory of this repository.

### Dependencies
This Python package requires version 3.7 or higher and the Python packages,

```
numpy
scipy
bokeh [required for only for plotting usage examples]
```

This software has been tested on Python 3.9.

## Installation
To install `nems_fingerprint` and dependent package `comsol_mesh` locally using the Python package manager `pip`, run the following commands in the command-line

```sh
git clone --depth 1 https://github.com/NEMS-AI/nems_fingerprint.git
git clone --depth 1 https://github.com/NEMS-AI/comsol_mesh.git
pip install -e nems_fingerprint
pip install -e comsol_mesh
```

Installation should take less than a minute.