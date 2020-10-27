#!/usr/bin/env python
# coding: utf-8

# 
# <div class="alert alert-info">
# 
# **This is a fixed-text formatted version of a Jupyter notebook**
# 
# - Try online [![Binder](https://static.mybinder.org/badge.svg)](https://mybinder.org/v2/gh/gammapy/gammapy-webpage/master?urlpath=lab/tree/analysis_3d.ipynb)
# - You can contribute with your own notebooks in this
# [GitHub repository](https://github.com/gammapy/gammapy/tree/master/docs/tutorials).
# - **Source files:**
# [analysis_3d.ipynb](../_static/notebooks/analysis_3d.ipynb) |
# [analysis_3d.py](../_static/notebooks/analysis_3d.py)
# </div>
# 

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
from scipy.stats import norm
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import (
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit
from gammapy.maps import Map
from gammapy.estimators import ExcessMapEstimator
from gammapy.datasets import MapDataset


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

# The FoV radius to use for cutouts
config.datasets.geom.selection.offset_max = 3.5 * u.deg
config.datasets.safe_mask.methods = ["aeff-default", "offset-max"]

# We now fix the energy axis for the counts map - (the reconstructed energy binning)
config.datasets.geom.axes.energy.min = "0.1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 10

# We now fix the energy axis for the IRF maps (exposure, etc) - (the true enery binning)
config.datasets.geom.axes.energy_true.min = "0.08 TeV"
config.datasets.geom.axes.energy_true.max = "12 TeV"
config.datasets.geom.axes.energy_true.nbins = 14


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


dataset_stacked = analysis_stacked.datasets["stacked"]
print(dataset_stacked)


# In[ ]:


# To plot a smooth counts map
dataset_stacked.counts.smooth(0.02 * u.deg).plot_interactive(add_cbar=True)


# In[ ]:


# And the background map
dataset_stacked.background_model.map.plot_interactive(add_cbar=True)


# In[ ]:


# You can also get an excess image with a few lines of code:
counts = dataset_stacked.counts.sum_over_axes()
background = dataset_stacked.background_model.map.sum_over_axes()
excess = counts - background
excess.smooth("0.06 deg").plot(stretch="sqrt", add_cbar=True);


# ### Modeling and fitting
# 
# Now comes the interesting part of the analysis - choosing appropriate models for our source and fitting them.
# 
# We choose a point source model with an exponential cutoff power-law spectrum.
# 
# To select a certain energy range for the fit we can create a fit mask:

# In[ ]:


coords = dataset_stacked.counts.geom.get_coord()
mask_energy = coords["energy"] > 0.3 * u.TeV
dataset_stacked.mask_fit = Map.from_geom(
    geom=dataset_stacked.counts.geom, data=mask_energy
)


# In[ ]:


spatial_model = PointSpatialModel(
    lon_0="-0.05 deg", lat_0="-0.05 deg", frame="galactic"
)
spectral_model = ExpCutoffPowerLawSpectralModel(
    index=2.3,
    amplitude=2.8e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1.0 * u.TeV,
    lambda_=0.02 / u.TeV,
)

model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
    name="gc-source",
)

dataset_stacked.models.append(model)
dataset_stacked.background_model.spectral_model.norm.value = 1.3


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit = Fit([dataset_stacked])\nresult = fit.run(optimize_opts={"print_level": 1})')


# ### Fit quality assesment and model residuals for a `MapDataset`

# We can access the results dictionary to see if the fit converged:

# In[ ]:


print(result)


# Check best-fit parameters and error estimates:

# In[ ]:


result.parameters.to_table()


# A quick way to inspect the model residuals is using the function `~MapDataset.plot_residuals()`. This function computes and plots a residual image (by default, the smoothing radius is `0.1 deg` and `method=diff`, which corresponds to a simple `data - model` plot):

# In[ ]:


dataset_stacked.plot_residuals(method="diff/sqrt(model)", vmin=-1, vmax=1)


# The same function can also extract and display spectral residuals, in case a region (used for the spectral extraction) is passed:

# In[ ]:


region = CircleSkyRegion(spatial_model.position, radius=0.15 * u.deg)

dataset_stacked.plot_residuals(
    method="diff/sqrt(model)", vmin=-1, vmax=1, region=region
)


# This way of accessing residuals is quick and handy, but comes with limitations. For example:
# - In case a fitting energy range was defined using a `MapDataset.mask_fit`, it won't be taken into account. Residuals will be summed up over the whole reconstructed energy range
# - In order to make a proper statistic treatment, instead of simple residuals a proper residuals significance map should be computed
# 
# A more accurate way to inspect spatial residuals is the following:

# In[ ]:


# TODO: clean this up
estimator = ExcessMapEstimator(
    correlation_radius="0.1 deg", selection_optional=[]
)
dataset_image = dataset_stacked.to_image()
estimator_dict = estimator.run(dataset_image)

residuals_significance = estimator_dict["sqrt_ts"]
residuals_significance.sum_over_axes().plot(cmap="coolwarm", add_cbar=True)


# Distribution of residuals significance in the full map geometry:

# In[ ]:


# TODO: clean this up
significance_data = residuals_significance.data

# #Remove bins that are inside an exclusion region, that would create an artificial peak at significance=0.
# #For now these lines are useless, because to_image() drops the mask fit
# mask_data = dataset_image.mask_fit.sum_over_axes().data
# excluded = mask_data == 0
# significance_data = significance_data[~excluded]
significance_data = significance_data.flatten()

plt.hist(significance_data, density=True, alpha=0.9, color="red", bins=30)
mu, std = norm.fit(significance_data)
x = np.linspace(
    np.min(significance_data) - 1, np.max(significance_data) + 1, 50
)
p = norm.pdf(x, mu, std)
plt.plot(
    x,
    p,
    lw=2,
    color="black",
    label=r"$\mu$ = {:.2f}, $\sigma$ = {:.2f}".format(mu, std),
)
plt.legend(fontsize=17)
plt.xlim(-6, 10)


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


# You can access each one by name or by index, eg:
print(analysis_joint.datasets[0])


# After the data reduction stage, it is nice to get a quick summary info on the datasets. 
# Here, we look at the statistics in the center of Map, by passing an appropriate `region`. To get info on the entire spatial map, omit the region argument.

# In[ ]:


analysis_joint.datasets.info_table()


# In[ ]:


# Add the model on each of the datasets
model_joint = model.copy(name="source-joint")
for dataset in analysis_joint.datasets:
    dataset.models.append(model_joint)
    dataset.background_model.spectral_model.norm.value = 1.1


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fit_joint = Fit(analysis_joint.datasets)\nresult_joint = fit_joint.run()')


# ### Fit quality assessment and model residuals for a joint `Datasets` 

# We can access the results dictionary to see if the fit converged:

# In[ ]:


print(result_joint)


# Check best-fit parameters and error estimates:

# In[ ]:


result_joint.parameters.to_table()


# The information on which parameter belongs to which dataset is not listed explicitly in the table (yet), but the order of parameters is conserved. You can always access the underlying object tree as well to get specific parameter values:

# In[ ]:


for dataset in analysis_joint.datasets:
    print(dataset.background_model.spectral_model.norm.value)


# Since the joint dataset is made of multiple datasets, we can either:
# - Look at the residuals for each dataset separately. In this case, we can directly refer to the section `Fit quality and model residuals for a MapDataset` in this notebook
# - Look at a stacked residual map. 

# In the latter case, we need to properly stack the joint dataset before computing the residuals:

# In[ ]:


# TODO: clean this up

# We need to stack on the full geometry, so we use to geom from the stacked counts map.
stacked = MapDataset.from_geoms(
    geom=dataset_stacked.counts.geom,
    geom_exposure=dataset_stacked.exposure.geom,
    geom_edisp=dataset_stacked.edisp.edisp_map.geom,
    geom_psf=dataset_stacked.psf.psf_map.geom,
)

for dataset in analysis_joint.datasets:
    # TODO: Apply mask_fit before stacking
    stacked.stack(dataset)


# Then, we can access the stacked model residuals as previously shown in the section `Fit quality and model residuals for a MapDataset` in this notebook.

# Finally, let us compare the spectral results from the stacked and joint fit:

# In[ ]:


def plot_spectrum(model, result, label, color):
    spec = model.spectral_model
    energy_range = [0.3, 10] * u.TeV
    spec.plot(
        energy_range=energy_range, energy_power=2, label=label, color=color
    )
    spec.plot_error(energy_range=energy_range, energy_power=2, color=color)


# In[ ]:


plot_spectrum(model, result, label="stacked", color="tab:blue")
plot_spectrum(model_joint, result_joint, label="joint", color="tab:orange")
plt.legend()


# ## Summary
# 
# Note that this notebook aims to show you the procedure of a 3D analysis using just a few observations. Results get much better for a more complete analysis considering the GPS dataset from the CTA First Data Challenge (DC-1) and also the CTA model for the Galactic diffuse emission, as shown in the next image:

# ![](images/DC1_3d.png)

# The complete tutorial notebook of this analysis is available to be downloaded in [GAMMAPY-EXTRA](https://github.com/gammapy/gammapy-extra) repository at https://github.com/gammapy/gammapy-extra/blob/master/analyses/cta_1dc_gc_3d.ipynb).

# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1 and add it to the combined model.
# * Perform modeling in more details - Add diffuse component, get flux points.
