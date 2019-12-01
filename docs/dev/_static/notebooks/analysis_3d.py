#!/usr/bin/env python
# coding: utf-8

# # 3D analysis
# 
# **TODO: Reduce by using HLI.**
# 
# This tutorial shows how to run a stacked 3D map-based analysis using three example observations of the Galactic center region with CTA.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.maps import WcsGeom, MapAxis
from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
from gammapy.modeling.models import (
    SkyModel,
    SkyModels,
    SkyDiffuseCube,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.spectrum import FluxPointsEstimator
from gammapy.modeling import Fit


# ## Prepare modeling input data
# 
# ### Prepare input maps
# 
# We first use the `~gammapy.data.DataStore` object to access the CTA observations and retrieve a list of observations by passing the observations IDs to the `~gammapy.data.DataStore.get_observations()` method:

# In[ ]:


# Define which data to use and print some information
data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")
data_store.info()
print("ONTIME (hours): ", data_store.obs_table["ONTIME"].sum() / 3600)
print("Observation table: ", data_store.obs_table.colnames)
print("HDU table: ", data_store.hdu_table.colnames)


# In[ ]:


# Select some observations from these dataset by hand
obs_ids = [110380, 111140, 111159]
observations = data_store.get_observations(obs_ids)


# Now we define a reference geometry for our analysis, We choose a WCS based gemoetry with a binsize of 0.02 deg and also define an energy axis: 

# In[ ]:


energy_axis = MapAxis.from_edges(
    np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    coordsys="GAL",
    proj="CAR",
    axes=[energy_axis],
)


# The `~gammapy.cube.MapDatasetMaker` object is initialized with this reference geometry and a field of view cut of 4 deg:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'stacked = MapDataset.create(geom=geom)\n\nmaker = MapDatasetMaker()\nmaker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)\n\nfor obs in observations:\n    cutout = stacked.cutout(obs.pointing_radec, width="8 deg")\n    dataset = maker.run(stacked, obs)\n    dataset = maker_safe_mask.run(dataset, obs)\n    stacked.stack(dataset)')


# This is what the stacked counts image looks like:

# In[ ]:


counts = stacked.counts.sum_over_axes()
counts.smooth(width="0.05 deg").plot(stretch="sqrt", add_cbar=True, vmax=6);


# This is the background image:

# In[ ]:


background = stacked.background_model.map.sum_over_axes()
background.plot(stretch="sqrt", add_cbar=True, vmax=6);


# And this one the exposure image:

# In[ ]:


exposure = stacked.exposure.sum_over_axes()
exposure.plot(stretch="sqrt", add_cbar=True);


# We can also compute an excess image just with  a few lines of code:

# In[ ]:


excess = counts - background
excess.smooth(5).plot(stretch="sqrt", add_cbar=True);


# In[ ]:


position = SkyCoord("0 deg", "0 deg", frame="galactic")
stacked.edisp = stacked.edisp.get_energy_dispersion(
    position=position, e_reco=energy_axis.edges
)


# In[ ]:


stacked.psf = stacked.psf.get_psf_kernel(
    position=position, geom=geom, max_radius="0.3 deg"
)


# ### Save dataset to disk
# 
# It is common to run the preparation step independent of the likelihood fit, because often the preparation of maps, PSF and energy dispersion is slow if you have a lot of data. We first create a folder:

# In[ ]:


path = Path("analysis_3d")
path.mkdir(exist_ok=True)


# And then write the maps and IRFs to disk by calling the dedicated `~gammapy.cube.MapDataset.write()` method:

# In[ ]:


filename = path / "stacked-dataset.fits.gz"
stacked.write(filename, overwrite=True)


# ## Likelihood fit
# 
# ### Reading the dataset
# As first step we read in the maps and IRFs that we have saved to disk again:

# In[ ]:


stacked = MapDataset.read(filename)


# ### Fit mask
# 
# To select a certain energy range for the fit we can create a fit mask:

# In[ ]:


coords = stacked.counts.geom.get_coord()
mask_energy = coords["energy"] > 0.3 * u.TeV
stacked.mask_safe.data &= mask_energy


# ### Model fit
# 
# No we are ready for the actual likelihood fit. We first define the model as a combination of a point source with a powerlaw:

# In[ ]:


spatial_model = PointSpatialModel(
    lon_0="0.01 deg", lat_0="0.01 deg", frame="galactic"
)
spectral_model = ExpCutoffPowerLawSpectralModel(
    index=2,
    amplitude=3e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1.0 * u.TeV,
    lambda_=0.1 / u.TeV,
)

model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="gc-source",
)
stacked.models = model

stacked.background_model.norm.value = 1.3


# No we run the model fit:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit([stacked])\nresult = fit.run(optimize_opts={"print_level": 1})')


# In[ ]:


result.parameters.to_table()


# ### Check model fit
# 
# We check the model fit by computing and plotting a residual image:

# In[ ]:


stacked.plot_residuals(method="diff/sqrt(model)", vmin=-1, vmax=1)


# We can also plot the best fit spectrum. For that need to extract the covariance of the spectral parameters.

# In[ ]:


spec = model.spectral_model

# set covariance on the spectral model
covar = result.parameters.get_subcovariance(spec.parameters)
spec.parameters.covariance = covar

energy_range = [0.3, 10] * u.TeV
spec.plot(energy_range=energy_range, energy_power=2)
spec.plot_error(energy_range=energy_range, energy_power=2)


# Apparently our model should be improved by adding a component for diffuse Galactic emission and at least one second point source.

# ### Add Galactic diffuse emission to model

# We use both models at the same time, our diffuse model (the same from the Fermi file used before) and our model for the central source. This time, in order to make it more realistic, we will consider an exponential cut off power law spectral model for the source. We will fit again the normalization and tilt of the background.

# In[ ]:


diffuse_model = SkyDiffuseCube.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
)


# In[ ]:


dataset_combined = stacked.copy()


# In[ ]:


dataset_combined.model = SkyModels([model, diffuse_model])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_combined = Fit([dataset_combined])\nresult_combined = fit_combined.run()')


# As we can see we have now two components in our model, and we can access them separately.

# In[ ]:


# Checking normalization value (the closer to 1 the better)
print(dataset_combined)


# You can see that the normalization of the background has vastly improved

# Just as a comparison, we can the previous residual map (top) and the new one (bottom) with the same scale:

# In[ ]:


stacked.plot_residuals(vmin=-1, vmax=1)
dataset_combined.plot_residuals(vmin=-1, vmax=1);


# ## Computing Flux Points
# 
# Finally we compute flux points for the galactic center source. For this we first define an energy binning:

# In[ ]:


e_edges = [0.3, 1, 3, 10] * u.TeV
fpe = FluxPointsEstimator(
    datasets=[dataset_combined], e_edges=e_edges, source="gc-source"
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'flux_points = fpe.run()')


# Now let's plot the best fit model and flux points:

# In[ ]:


flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(energy_power=2)
model.spectral_model.plot(ax=ax, energy_range=energy_range, energy_power=2);


# ## Summary
# 
# Note that this notebook aims to show you the procedure of a 3D analysis using just a few observations and a cutted Fermi model. Results get much better for a more complete analysis considering the GPS dataset from the CTA First Data Challenge (DC-1) and also the CTA model for the Galactic diffuse emission, as shown in the next image:

# ![](images/DC1_3d.png)

# The complete tutorial notebook of this analysis is available to be downloaded in [GAMMAPY-EXTRA](https://github.com/gammapy/gammapy-extra) repository at https://github.com/gammapy/gammapy-extra/blob/master/analyses/cta_1dc_gc_3d.ipynb).

# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1 and add it to the combined model.

# In[ ]:




