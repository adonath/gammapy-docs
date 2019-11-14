#!/usr/bin/env python
# coding: utf-8

# # Joint Crab analysis with 1D and 3D datasets

# This tutorial illustrates how to run a joint analysis with different datasets.
# We look at the gamma-ray emission from the Crab nebula between 10 GeV and 100 TeV.
# The spectral parameters are optimized by combining a 3D analysis of Fermi-LAT data, a ON/OFF spectral analysis of HESS data, and flux points from HAWC.  
# 
# 
# 

# In[ ]:


import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling import Fit, Datasets
from gammapy.spectrum import (
    FluxPoints,
    FluxPointsEstimator,
    FluxPointsDataset,
    SpectrumDatasetOnOff,
)
from pathlib import Path


# ## Data and models files
# 
# We are going to use pre-made datasets. Links toward other tutorials detailing on how to prepare datasets are given at the end.
# The datasets serialization produce YAML files listing the datasets and models. In the following cells we show an example containning only the Fermi-LAT dataset and the Crab model.
# 
# Fermi-LAT-3FHL_datasets.yaml:
datasets:
- name: Fermi-LAT
  type: MapDataset
  likelihood: cash
  models:
- Crab Nebula
  background: background
  filename: $GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_data_Fermi-LAT.fits
# Fermi-LAT-3FHL_models.yaml:
components:
- name: Crab Nebula
  type: SkyModel
  spatial:
    type: PointSpatialModel
    frame: icrs
    parameters:
    - name: lon_0
      value: 83.63310241699219
      unit: deg
      min: .nan
      max: .nan
      frozen: true
    - name: lat_0
      value: 22.019899368286133
      unit: deg
      min: -90.0
      max: 90.0
      frozen: true
  spectral:
    type: LogParabolaSpectralModel
    parameters:
    - name: amplitude
      value: 0.3415498620816483
      unit: cm-2 s-1 TeV-1
      min: .nan
      max: .nan
      frozen: false
    - name: reference
      value: 5.054833602905273e-05
      unit: TeV
      min: .nan
      max: .nan
      frozen: true
    - name: alpha
      value: 2.510798031388936
      unit: ''
      min: .nan
      max: .nan
      frozen: false
    - name: beta
      value: -0.022476498188855533
      unit: ''
      min: .nan
      max: .nan
      frozen: false
- name: background
  type: BackgroundModel
  parameters:
  - name: norm
    value: 0.9544383244743555
    unit: ''
    min: 0.0
    max: .nan
    frozen: false
  - name: tilt
    value: 0.0
    unit: ''
    min: .nan
    max: .nan
    frozen: true
  - name: reference
    value: 1.0
    unit: TeV
    min: .nan
    max: .nan
    frozen: true

# ## Reading  different datasets
# 
# 
# ### Fermi-LAT 3FHL: map dataset for 3D analysis
# For now we let's use the datasets serialization only to read the 3D `MapDataset` associated to Fermi-LAT 3FHL data and models.

# In[ ]:


path = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL"
filedata = Path(path + "_datasets.yaml")
filemodel = Path(path + "_models.yaml")
datasets = Datasets.from_yaml(filedata=filedata, filemodel=filemodel)
dataset_fermi = datasets[0]


# We get the Crab spectral model in order to share it with the other datasets

# In[ ]:


crab_spec = [
    model.spectral_model
    for model in dataset_fermi.model
    if model.name == "Crab Nebula"
][0]

print(crab_spec)


# ### HESS-DL3: 1D ON/OFF dataset for spectral fitting
# 
# The ON/OFF datasets can be read from PHA files following the [OGIP standards](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html).
# We read the PHA files from each observation, and compute a stacked dataset for simplicity.
# Then the Crab spectral model previously defined is added to the dataset.

# In[ ]:


obs_ids = [23523, 23526]
datasets = []
for obs_id in obs_ids:
    dataset = SpectrumDatasetOnOff.from_ogip_files(
        f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
    )
    datasets.append(dataset)
dataset_hess = Datasets(datasets).stack_reduce()
dataset_hess.name = "HESS"
dataset_hess.model = crab_spec


# ### HAWC: 1D dataset for flux point fitting
# 
# The HAWC flux point are taken from https://arxiv.org/pdf/1905.12518.pdf. Then these flux points are read from a pre-made FITS file and passed to a `FluxPointsDataset` together with the source spectral model.
# 

# In[ ]:


# read flux points from https://arxiv.org/pdf/1905.12518.pdf
filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
flux_points_hawc = FluxPoints.read(filename)
dataset_hawc = FluxPointsDataset(crab_spec, flux_points_hawc, name="HAWC")


# ## Datasets serialization
# 
# The `datasets` object contains each dataset previously defined. 
# It can be saved on disk as datasets.yaml, models.yaml, and several data files specific to each dataset. Then the `datasets` can be rebuild later from these files.

# In[ ]:


datasets = Datasets([dataset_fermi, dataset_hess, dataset_hawc])
path = Path("crab-3datasets")
# datasets.to_yaml(path=path, prefix="crab_10GeV_100TeV", overwrite=True)
path.mkdir(exist_ok=True)
filedata = path / "crab_10GeV_100TeV_datasets.yaml"
filemodel = path / "crab_10GeV_100TeV_models.yaml"
# datasets.from_yaml(filedata=filedata, filemodel=filemodel)


# ## Joint analysis
# 
# We run the fit on the `Datasets` object that include a dataset for each instrument
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_joint = Fit(datasets)\nresults_joint = fit_joint.run()\nprint(results_joint)\nprint(results_joint.parameters.to_table())')


# Let's display only the parameters of the Crab spectral model

# In[ ]:


crab_spec.parameters.covariance = results_joint.parameters.get_subcovariance(
    crab_spec.parameters
)
print(crab_spec)


# We can compute flux points for Fermi-LAT and HESS datasets in order plot them together with the HAWC flux point.

# In[ ]:


# compute Fermi-LAT and HESS flux points
e_min, e_max = 0.01, 2.0
El_fermi = np.logspace(np.log10(e_min), np.log10(e_max), 6) * u.TeV
flux_points_fermi = FluxPointsEstimator(
    datasets=[dataset_fermi], e_edges=El_fermi, source="Crab Nebula"
).run()

e_min, e_max = 1.0, 15.0
El_hess = np.logspace(np.log10(e_min), np.log10(e_max), 6) * u.TeV
flux_points_hess = FluxPointsEstimator(
    datasets=[dataset_hess], e_edges=El_hess
).run()


# Now, Let's plot the Crab spectrum fitted and the flux points of each instrument.
# 

# In[ ]:


# display spectrum and flux points
energy_range = [0.01, 120] * u.TeV
plt.figure(figsize=(8, 6))
ax = crab_spec.plot(energy_range=energy_range, energy_power=2, label="Model")
crab_spec.plot_error(ax=ax, energy_range=energy_range, energy_power=2)
flux_points_fermi.plot(ax=ax, energy_power=2, label="Fermi-LAT")
flux_points_hess.plot(ax=ax, energy_power=2, label="HESS")
flux_points_hawc.plot(ax=ax, energy_power=2, label="HAWC")
plt.legend();


# ## To go further
# 
# More details on how to prepare datasets with the high and low level interfaces are available in these tutorials: 
# - https://docs.gammapy.org/0.14/notebooks/fermi_lat.html
# - https://docs.gammapy.org/dev/notebooks/hess.html
# - https://docs.gammapy.org/dev/notebooks/spectrum_analysis.html
# 

# In[ ]:




