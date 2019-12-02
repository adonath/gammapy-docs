#!/usr/bin/env python
# coding: utf-8

# # 3D analysis
# 
# This tutorial does a 3D map based analsis on the galactic center, using simulated observations from the CTA-1DC. We will use the high level interface for the data reduction, and then do a detailed modelling. This will be done in two different ways:
# 
# - stacking all the maps together and fitting the stacked maps
# - handling all the observations separately and doing a joint fitting on all the maps

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from pathlib import Path
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import (
    SkyModel,
    SkyModels,
    SkyDiffuseCube,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit
from gammapy.spectrum import FluxPointsEstimator


# ## Analysis configuration

# In this section we select observations and define the analysis geometries, irrespective of  joint/stacked analysis. For configuration of the analysis, we will programatically build a config file from scratch.

# In[ ]:


config = AnalysisConfig()
# The config file is now empty, with only a few defaults specified.
print(config)


# In[ ]:


# Selecting the observations
config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
config.observations.obs_ids = [110380, 111140, 111159]


# In[ ]:


# Defining a reference geometry for the reduced datasets

config.datasets.type = "3d"  # Analysis type is 3D

config.datasets.geom.wcs.skydir = {
    "lon": "0 deg",
    "lat": "0 deg",
    "frame": "galactic",
}  # The WCS geometry - centered on the galactic center
config.datasets.geom.wcs.fov = {"width": "10 deg", "height": "8 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# The FoV offset cut
config.datasets.geom.selection.offset_max = 4.0 * u.deg

# We now fix the energy axis for the counts map - (the reconstructed energy binning)
config.datasets.geom.axes.energy.min = "0.1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 10

# We now fix the energy axis for the IRF maps (exposure, etc) - (the true enery binning)
config.datasets.geom.axes.energy_true.min = "0.02 TeV"
config.datasets.geom.axes.energy_true.max = "20 TeV"
config.datasets.geom.axes.energy_true.nbins = 20


# In[ ]:


print(config)


# ## Configuration for stacked and joint analysis
# 
# This is done just by specfiying the flag on `config.datasets.stack`. Since the internal machinery will work differently for the two cases, we will write it as two config files and save it to disc in YAML format for future reference. 

# In[ ]:


config_stack = config.copy(deep=True)
config_stack.datasets.stack = True

config_joint = config.copy(deep=True)
config_joint.datasets.stack = False


# In[ ]:


# To prevent unnecessary cluttering, we write it in a separate folder.
path = Path("analysis_3d")
path.mkdir(exist_ok=True)
config_joint.write(path=path / "config_joint.yaml", overwrite=True)
config_stack.write(path=path / "config_stack.yaml", overwrite=True)


# ## Stacked analysis
# 
# ### Data reduction
# 
# We first show the steps for the stacked analysis and then repeat the same for the joint analysis later
# 

# In[ ]:


# Reading yaml file:
config_stacked = AnalysisConfig.read(path=path / "config_stack.yaml")


# In[ ]:


analysis_stacked = Analysis(config_stacked)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# select observations:\nanalysis_stacked.get_observations()\n\n# run data reduction\nanalysis_stacked.get_datasets()')


# We have one final dataset, which you can print and explore

# In[ ]:


print(analysis_stacked.datasets)


# In[ ]:


print(analysis_stacked.datasets["stacked"])


# In[ ]:


# To plot a smooth counts map
analysis_stacked.datasets["stacked"].counts.smooth(
    0.02 * u.deg
).plot_interactive(add_cbar=True)


# In[ ]:


# And the background map
analysis_stacked.datasets["stacked"].background_model.map.smooth(
    0.02 * u.deg
).plot_interactive(add_cbar=True)


# In[ ]:


# You can also get an excess image with a few lines of code:
counts = analysis_stacked.datasets["stacked"].counts.sum_over_axes()
background = analysis_stacked.datasets[
    "stacked"
].background_model.map.sum_over_axes()
excess = counts - background
excess.smooth(5).plot(stretch="sqrt", add_cbar=True);


# ### Modeling and fitting
# 
# Now comes the interesting part of the analysis - choosing appropriate models for our source and fitting them.
# 
# We choose a point source model with an exponential cutoff power-law spectrum.
# 
# To select a certain energy range for the fit we can create a fit mask:

# In[ ]:


coords = analysis_stacked.datasets["stacked"].counts.geom.get_coord()
mask_energy = coords["energy"] > 0.3 * u.TeV
analysis_stacked.datasets["stacked"].mask_safe.data &= mask_energy


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

sky_model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="gc-source",
)
model = sky_model.copy()
analysis_stacked.datasets["stacked"].models = model

analysis_stacked.datasets["stacked"].background_model.norm.value = 1.3


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit(analysis_stacked.datasets)\nresult = fit.run(optimize_opts={"print_level": 1})')


# In[ ]:


result.parameters.to_table()


# ### Check model fit
# 
# We check the model fit by computing and plotting a residual image:

# In[ ]:


analysis_stacked.datasets["stacked"].plot_residuals(
    method="diff/sqrt(model)", vmin=-1, vmax=1
)


# We can also plot the best fit spectrum. For that need to extract the covariance of the spectral parameters.

# In[ ]:


spec = model.spectral_model

# set covariance on the spectral model
covar = result.parameters.get_subcovariance(spec.parameters)
spec.parameters.covariance = covar

energy_range = [0.3, 10] * u.TeV
spec.plot(energy_range=energy_range, energy_power=2)
spec.plot_error(energy_range=energy_range, energy_power=2)


# The high value of the background normalisation `norm = 1.24` suggests that our model should be improved by adding a component for diffuse Galactic emission and at least one second point source.

# ### Galactic diffuse emission
# 
# We use both models at the same time, our diffuse model (from the Fermi diffuse model) and our model for the central source. This time, in order to make it more realistic, we will consider an exponential cut off power law spectral model for the source. We will fit again the normalization and tilt of the background.

# In[ ]:


diffuse_model = SkyDiffuseCube.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
)


# In[ ]:


dataset_stacked = analysis_stacked.datasets["stacked"].copy()


# In[ ]:


dataset_stacked.models = SkyModels([model, diffuse_model])


# In[ ]:


# As we can see we have now two components in our model, and we can access them separately.
print(dataset_stacked)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_combined = Fit([dataset_stacked])\nresult_combined = fit_combined.run()')


# In[ ]:


result_combined.parameters.to_table()


# You can see that the normalization of the background has vastly improved

# Just as a comparison, we can the previous residual map (top) and the new one (bottom) with the same scale:

# In[ ]:


analysis_stacked.datasets["stacked"].plot_residuals(vmin=-1, vmax=1)
dataset_stacked.plot_residuals(vmin=-1, vmax=1);


# ### Flux points
# 
# Finally we compute flux points for the galactic center source. For this we first define an energy binning:

# In[ ]:


e_edges = [0.3, 1, 3, 10] * u.TeV
fpe = FluxPointsEstimator(
    datasets=[dataset_stacked], e_edges=e_edges, source="gc-source-copy"
)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'flux_points = fpe.run()')


# Now let's plot the best fit model and flux points:

# In[ ]:


flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(energy_power=2)
model.spectral_model.plot(ax=ax, energy_range=energy_range, energy_power=2);


# ## Joint analysis
# 
# In this section, we perform a joint analysis of the same data. Of course, joint fitting is considerably heavier than stacked one, and should always be handled with care. For brevity, we only show the analysis for a point source fitting without re-adding a diffuse component again. 
# 
# ### Data reduction

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Read the yaml file from disk\nconfig_joint = AnalysisConfig.read(path=path / "config_joint.yaml")\nanalysis_joint = Analysis(config_joint)\n\n# select observations:\nanalysis_joint.get_observations()\n\n# run data reduction\nanalysis_joint.get_datasets()')


# In[ ]:


# You can see there are 3 datasets now
print(analysis_joint.datasets)


# In[ ]:


# You can access each one by its name, eg:
print(analysis_joint.datasets["obs_110380"])


# In[ ]:


# Add the model on each of the datasets
model = sky_model.copy()
for dataset in analysis_joint.datasets:
    dataset.models = model


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_joint = Fit(analysis_joint.datasets)\nresult_joint = fit_joint.run()')


# In[ ]:


print(result)


# In[ ]:


fit_joint.datasets.parameters.to_table()


# The information which parameter belongs to which dataset is not listed explicitly in the table (yet), but the order of parameters is conserved. You can always access the underlying object tree as well to get specific parameter values:

# In[ ]:


for dataset in analysis_joint.datasets:
    print(dataset.background_model.norm.value)


# ### Residuals
# 
# Since we have multiple datasets, we can either look at a stacked residual map, or the residuals for each dataset. Each `gammapy.cube.MapDataset` object is equipped with a method called `gammapy.cube.MapDataset.plot_residuals()`, which displays the spatial and spectral residuals (computed as *counts-model*) for the dataset. Optionally, these can be normalized as *(counts-model)/model* or *(counts-model)/sqrt(model)*, by passing the parameter `norm='model` or `norm=sqrt_model`.

# In[ ]:


# To see the spectral residuals, we have to define a region for the spectral extraction
region = CircleSkyRegion(spatial_model.position, radius=0.15 * u.deg)


# In[ ]:


for dataset in analysis_joint.datasets:
    ax_image, ax_spec = dataset.plot_residuals(
        region=region, vmin=-0.5, vmax=0.5, method="diff"
    )


# In[ ]:


from gammapy.maps import Map

# We need to stack on the full geometry, so we use to geom from the stacked counts map.
residuals_stacked = Map.from_geom(analysis_stacked.datasets[0].counts.geom)

for dataset in analysis_joint.datasets:
    residuals = dataset.residuals()
    residuals_stacked.stack(residuals)

    residuals_stacked.sum_over_axes().smooth("0.08 deg").plot(
        vmin=-1, vmax=1, cmap="coolwarm", add_cbar=True, stretch="linear"
    );


# ## Summary
# 
# Note that this notebook aims to show you the procedure of a 3D analysis using just a few observations and a cutted Fermi model. Results get much better for a more complete analysis considering the GPS dataset from the CTA First Data Challenge (DC-1) and also the CTA model for the Galactic diffuse emission, as shown in the next image:

# ![](images/DC1_3d.png)

# The complete tutorial notebook of this analysis is available to be downloaded in [GAMMAPY-EXTRA](https://github.com/gammapy/gammapy-extra) repository at https://github.com/gammapy/gammapy-extra/blob/master/analyses/cta_1dc_gc_3d.ipynb).

# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1 and add it to the combined model.
# * Perform joint fit in more details - Add diffuse component, get flux points.
