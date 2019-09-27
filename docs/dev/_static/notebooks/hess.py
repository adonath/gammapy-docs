#!/usr/bin/env python
# coding: utf-8

# # H.E.S.S. with Gammapy
# 
# This tutorial explains how to analyse [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS) data with Gammapy.
# 
# We will analyse four observation runs of the Crab nebula, which are part of the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/). In this tutorial we will make an image and a spectrum. The [light_curve.ipynb](light_curve.ipynb) notbook contains an example how to make a light curve.
# 
# To do a 3D analysis, one needs to do a 3D background estimate. In [background_model.ipynb](background_model.ipynb) we have started to make a background model, and in this notebook we have a first look at a 3D analysis. But the results aren't OK yet, the background model needs to be improved. In this analysis, we also don't use the energy dispersion IRF yet, and we only analyse the data in the 1 TeV to 10 TeV range. The H.E.S.S. data was only released very recently, and 3D analysis in Gammapy is new. This tutorial will be improved soon.
# 
# This tutorial also shows how to do a classical image analysis using the ring bakground. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


import yaml
from pathlib import Path
import numpy as np
from scipy.stats import norm
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Tophat2DKernel
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.cube import (
    MapMaker,
    MapDataset,
    PSFKernel,
    MapMakerRing,
    RingBackgroundEstimator,
)
from gammapy.modeling.models import SkyModel, BackgroundModel, SkyModels
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.modeling.models import create_crab_spectral_model
from gammapy.modeling.models import PointSpatialModel
from gammapy.detect import compute_lima_on_off_image
from gammapy.scripts import Analysis
from gammapy.modeling import Fit


# ## Data access
# 
# To access the data, we use the `DataStore`, and we use the ``obs_table`` to select the Crab runs.

# In[ ]:


data_store = DataStore.from_file(
    "$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz"
)
mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_table = data_store.obs_table[mask]
observations = data_store.get_observations(obs_table["OBS_ID"])


# In[ ]:


# pos_crab = SkyCoord.from_name('Crab')
pos_crab = SkyCoord(83.633, 22.014, unit="deg")


# ## Maps
# 
# Let's make some 3D cubes, as well as 2D images.
# 
# For the energy, we make 5 bins from 1 TeV to 10 TeV.

# In[ ]:


energy_axis = MapAxis.from_edges(
    np.logspace(0, 1.0, 5), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(83.633, 22.014),
    binsz=0.02,
    width=(5, 5),
    coordsys="CEL",
    proj="TAN",
    axes=[energy_axis],
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, offset_max="2.5 deg")\nmaps = maker.run(observations)\nimages = maker.run_images()')


# In[ ]:


maps.keys()


# In[ ]:


images["counts"].smooth(3).plot(stretch="sqrt", vmax=2);


# ## PSF
# 
# Compute the mean PSF for these observations at the Crab position.

# In[ ]:


from gammapy.irf import make_mean_psf

table_psf = make_mean_psf(observations, pos_crab)


# In[ ]:


psf_kernel = PSFKernel.from_table_psf(table_psf, geom, max_radius="0.3 deg")
psf_kernel_array = psf_kernel.psf_kernel_map.sum_over_axes().data
# psf_kernel.psf_kernel_map.slice_by_idx({'energy': 0}).plot()
# plt.imshow(psf_kernel_array)


# ## Map fit
# 
# Let's fit this source assuming a Gaussian spatial shape and a power-law spectral shape, and a background with a flexible normalisation

# In[ ]:


spatial_model = PointSpatialModel(
    lon_0="83.6 deg", lat_0="22.0 deg", frame="icrs"
)
spectral_model = PowerLawSpectralModel(
    index=2.6, amplitude="5e-11 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)
background_model = BackgroundModel(maps["background"], norm=1.0)
background_model.parameters["tilt"].frozen = False


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dataset = MapDataset(\n    model=model,\n    counts=maps["counts"],\n    exposure=maps["exposure"],\n    background_model=background_model,\n    psf=psf_kernel,\n)\nfit = Fit(dataset)\nresult = fit.run()\nprint(result)')


# Best fit parameters:

# In[ ]:


result.parameters.to_table()


# Parameters covariance:

# In[ ]:


result.parameters.covariance_to_table()


# ## Residual image
# 
# We compute a residual image as `residual = counts - model`. Note that this is counts per pixel and our pixel size is 0.02 deg. Smoothing is counts-preserving. The residual image shows that currently both the source and the background modeling isn't very good. The background model is underestimated (so residual is positive), and the source model is overestimated.

# In[ ]:


npred = dataset.npred()
residual = Map.from_geom(maps["counts"].geom)
residual.data = maps["counts"].data - npred.data


# In[ ]:


residual.sum_over_axes().smooth("0.1 deg").plot(
    cmap="coolwarm", vmin=-0.2, vmax=0.2, add_cbar=True
);


# ## Spectrum
# 
# We could try to improve the background modeling and spatial model of the source. But let's instead turn to one of the classic IACT analysis techniques: use a circular on region and reflected regions for background estimation, and derive a spectrum for the source without having to assume a spatial model, or without needing a 3D background model.
# 
# We will use the high-level interface with the following configuration:

# In[ ]:


config = """
general:
    logging:
        level: INFO
    outdir: .

observations:
  datastore: $GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz
  filters:
  - filter_type: par_value
    variable: TARGET_NAME
    value_param: Crab

reduction:
    background:
        background_estimator: reflected
    containment_correction: true
    dataset-type: SpectrumDatasetOnOff
    stack-datasets: false
    geom:
        region:
          center:
            - 83.633 deg
            - 22.014 deg
          frame: icrs
          radius: 0.11 deg
        axes:
          - name: energy
            hi_bnd: 100
            lo_bnd: 0.01
            nbin: 73
            interp: log
            node_type: edges
            unit: TeV

model: model.yaml

fit:
    fit_range:
        min: 1 TeV
        max: 10 TeV
flux:
    fp_binning:
        lo_bnd: 1
        hi_bnd: 10
        nbin: 10
        unit: TeV
        interp: log      
"""
filename = Path("config.yaml")
filename.write_text(config)


# In[ ]:


model_config = """
components:
- name: source
  type: SkyModel
  spatial:
    type: PointSpatialModel
    frame: icrs
    parameters:
    - name: lon_0
      value: 83.633
      unit: deg
      frozen: true
    - name: lat_0 
      value: 22.14    
      unit: deg
      frozen: true
  spectral:
    type: PowerLawSpectralModel
    parameters:
    - name: amplitude      
      value: 1.0e-12
      unit: cm-2 s-1 TeV-1
    - name: index
      value: 2.0
      unit: ''
    - name: reference
      value: 1.0
      unit: TeV
      frozen: true
"""
filename = Path("model.yaml")
filename.write_text(model_config)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'analysis = Analysis.from_yaml("config.yaml")\nanalysis.get_observations()\nanalysis.get_datasets()\nanalysis.get_model()\nanalysis.run_fit()\nanalysis.get_flux_points()')


# In[ ]:


analysis.flux_points_dataset.model.parameters.covariance = (
    analysis.fit_result.parameters.covariance
)
print(analysis.flux_points_dataset.model)


# In[ ]:


plt.figure(figsize=(10, 8))
crab_ref = create_crab_spectral_model("hess_pl")

dataset_fp = analysis.flux_points_dataset

plot_kwargs = {
    "energy_range": [1, 10] * u.TeV,
    "flux_unit": "erg-1 cm-2 s-1",
    "energy_power": 2,
}

model_kwargs = {"label": "1D best fit model"}
model_kwargs.update(plot_kwargs)
ax_spectrum, ax_residuals = dataset_fp.peek(model_kwargs=model_kwargs)

crab_ref.plot(ax=ax_spectrum, label="H.E.S.S. 2006 PWL", **plot_kwargs)
model.spectral_model.plot(
    ax=ax_spectrum, label="3D best fit model", **plot_kwargs
)

ax_spectrum.set_ylim(1e-11, 1e-10)
ax_spectrum.legend();


# Again: please note that this tutorial notebook was put together quickly, the results obtained here are very preliminary. We will work on Gammapy and the analysis of data from the H.E.S.S. test release and update this tutorial soon.

# ## Exercises
# 
# - Try analysing another source, e.g. RX J1713.7−3946
# - Try another model, e.g. a Gaussian spatial shape or exponential cutoff power-law spectrum.

# In[ ]:




