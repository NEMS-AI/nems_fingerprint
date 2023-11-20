# NEMS Fingerprint
A Python package for mass spectrometry via Nano-electromechanical Systems (NEMS) using the **fingerprint method**.

## Description
The **fingerprint method** for NEMS mass spectrometry allows the mass of nanoscale analytes to be recovered from the frequency shifts induced in each oscillation mode upon absorption. By calibrating with analytes of known mass in the _learning phase_ the fingerprint method can develop a _fingerprint database_ which characterizes the response of a NEMS device. The masses of arbitrary analytes absorbing to the NEMS device can then be determined from frequency shifts in the _measurement phase_.

The **fingerprint method** provides an alternative to previous NEMS mass spectrometry methods which rely on simplified analytic models or finite element simulations of the oscillatory modes of the NEMS device. By removing the requirement for an underlying model of the oscillatory modes and calibrating on the device directly, the fingerprint method extends NEMS mass spectrometry to devices with arbitrary uncharacterized modes.

## Usage
The masses of analytes can be predicted using the known masses and frequency shifts of the learning phase.

```python
from nems_fingerprint import *

# Learning phase
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

## Installation
This Python package requires version 3.7 or higher. To install locally using the Python package manager `pip`, run the following commands in the command-line,

```sh
git clone --depth 1 https://github.com/NEMS-AI/nems_fingerprint.git
cd nems_fingerprint
pip install -e .
```

To install using the `conda` environment manager run,

```sh
git clone --depth 1 https://github.com/NEMS-AI/nems_fingerprint.git
cd nems_fingerprint
conda develop .
```
