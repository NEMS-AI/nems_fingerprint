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
# Load COMSOL mesh
comsol_objs = COMSOLObjects.from_file('data/shear_device.mphtxt')
cobj = comsol_objs[0]
mesh = Mesh.from_comsol_obj(cobj)

# Load device eigenmodes
cemodes = COMSOLEigenmodes.from_file('data/shear_device_eigenmodes.csv')
modes_field = Field.from_comsol_field(mesh, cemodes)

# Load top surface
surfaces = surfaces_from_comsol_obj(mesh, cobj)
top_surface = surfaces[-1]

# Define simulation
M0 = 1.0
mode_indices = [0, 1, 2, 3]
comsol_sim = COMSOLSimulation(
    mesh=mesh, 
    surface=top_surface, 
    modes_field=modes_field, 
    mode_idxs=mode_indices,
    mass_dist=Distribution.constant(M0),
    noise_dist=Distribution.constant(0.0)
)

# Learning phase
learning_events = comsol_sim.sample(n_events=10000)
mass_predictor = MassPredictNN(learning_events)

# Measurement phase
measurement_phase_shifts = comsol_sim.sample(n_events=2000).freq_shifts
predicted_masses = mass_predictor(measurement_phase_shifts)
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