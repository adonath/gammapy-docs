#!/usr/bin/env python
# coding: utf-8

# # First analysis
# 
# **TODO: Rewrite this notebook to be the "first analysis" one as described in PIG 18. Note that there is now a separate ``hess.ipynb`` that can be referenced from here, for a description of H.E.S.S. and the datasets. Also note that there is a follow-up `analysis2.ipynb`, this notebook can be short.**
# 
# In September 2018 the [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS) collaboration released a small subset of archival data in FITS format. This tutorial explains how to analyse this data with Gammapy. We will analyse four observation runs of the Crab nebula, which are part of the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/). The data was release without corresponding background models. In [background_model.ipynb](background_model.ipynb) we show how to make a simple background model, which is also used in this tutorial. The background model is not perfect; it assumes radial symmetry and is in general derived from only a few observations, but still good enough for a reliable analysis > 1TeV.
# 
# **Note:** The high level `Analysis` class is a new feature added in Gammapy v0.14. In the current state it supports the standard analysis cases of a joint or stacked 3D and 1D analysis. It provides only limited access to analaysis parameters via the config file. It is expected that the format of the YAML config will be extended and change in future Gammapy versions.
# 
# We will first show how to configure and run a stacked 3D analysis and then address the classical spectral analysis using reflected regions later. The structure of the tutorial follows a typical analysis:
# 
# - Analysis configuration
# - Observation slection
# - Data reduction
# - Model fitting
# - Estimating flux points
# 
# Finally we will compare the results against a reference model.

# ## Setup

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from regions import CircleSkyRegion
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model


# ## Analysis configuration
# 
# For configuration of the analysis we use the [YAML](https://en.wikipedia.org/wiki/YAML) data format. YAML is a machine readable serialisation format, that is also friendly for humans to read. In this tutorial we will write the configuration file just using Python strings, but of course the file can be created and modified with any text editor of your choice.
# 
# Here is what the configuration for our analysis looks like:

# In[ ]:


config_str = """
general:
    log:
        level: info
    outdir: .
observations:
    datastore: $GAMMAPY_DATA/hess-dl3-dr1/
    obs_cone: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 5 deg}
datasets:
    type: 3d
    stack: true
    geom:
        wcs:
            skydir: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg}
            binsize: 0.04 deg
            fov: {width: 5 deg, height: 5 deg}
            binsize_irf: 0.2 deg
            margin_irf: 0.5 deg
        selection:
            offset_max: 2.5 deg
        axes:
            energy: {min: 1 TeV, max: 10 TeV, nbins: 4}
            energy_true: {min: 1 TeV, max: 10 TeV, nbins: 5}            
fit:
    fit_range: {min: 1 TeV, max: 30 TeV}

flux_points:
    energy: {min: 1 TeV, max: 10 TeV, nbins: 3}
"""


# We first create an `~gammapy.analysis.AnalysisConfig` object from it:

# In[ ]:


config = AnalysisConfig.from_yaml(config_str)


# ##  Observation selection
# 
# Now we create the high level `~gammapy.analysis.Analysis` object from the config object:

# In[ ]:


analysis = Analysis(config)


# And directly select and load the observations from disk using `~gammapy.analysis.Analysis.get_observations()`:

# In[ ]:


analysis.get_observations()


# The observations are now available on the `Analysis` object. The selection corresponds to the following ids:

# In[ ]:


analysis.observations.ids


# Now we can access and inspect individual observations by accessing with the observation id:

# In[ ]:


print(analysis.observations["23592"])


# And also show a few overview plots using the `.peek()` method:

# In[ ]:


analysis.observations["23592"].peek()


# ## Data reduction
# 
# Now we proceed to the data reduction. In the config file we have chosen a WCS map geometry, energy axis and decided to stack the maps. We can run the reduction using `.get_datasets()`:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'analysis.get_datasets()')


# As we have chosen to stack the data, there is finally one dataset contained:

# In[ ]:


print(analysis.datasets)


# We can print the dataset as well:

# In[ ]:


print(analysis.datasets["stacked"])


# As you can see the dataset comes with a predefined background model out of the data reduction, but no source model has been set yet.
# 
# The counts, exposure and background model maps are directly available on the dataset and can be printed and plotted:

# In[ ]:


counts = analysis.datasets["stacked"].counts


# In[ ]:


print(counts)


# In[ ]:


counts.smooth("0.05 deg").plot_interactive()


# ## Model fitting
# 
# Now we define a model to be fitted to the dataset:

# In[ ]:


model_config = """
components:
- name: crab
  type: SkyModel
  spatial:
    type: PointSpatialModel
    frame: icrs
    parameters:
    - name: lon_0
      value: 83.63
      unit: deg
    - name: lat_0 
      value: 22.14    
      unit: deg
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


# Now we set the model on the analysis object:

# In[ ]:


analysis.set_model(model_config)


# In[ ]:


print(analysis.model)


# In[ ]:


print(analysis.model["crab"])


# Finally we run the fit:

# In[ ]:


analysis.run_fit()


# In[ ]:


print(analysis.fit_result)


# This is how we can write the model back to file again:

# In[ ]:


analysis.model.to_yaml("model-best-fit.yaml")


# In[ ]:


get_ipython().system('cat model-best-fit.yaml')


# ### Inspecting residuals
# 
# For any fit it is usefull to inspect the residual images. We have a few option on the dataset object to handle this. First we can use `.plot_residuals()` to plot a residual image, summed over all energies: 

# In[ ]:


analysis.datasets["stacked"].plot_residuals(
    method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
);


# In addition we can aslo specify a region in the map to show the spectral residuals:

# In[ ]:


region = CircleSkyRegion(
    center=SkyCoord("83.63 deg", "22.14 deg"), radius=0.5 * u.deg
)


# In[ ]:


analysis.datasets["stacked"].plot_residuals(
    region=region, method="diff/sqrt(model)", vmin=-0.5, vmax=0.5
);


# We can also directly access the `.residuals()` to get a map, that we can plot interactively:

# In[ ]:


residuals = analysis.datasets["stacked"].residuals(method="diff")
residuals.smooth("0.08 deg").plot_interactive(
    cmap="coolwarm", vmin=-0.1, vmax=0.1, stretch="linear", add_cbar=True
)


# ### Inspecting fit statistic profiles
# 
# To check the quality of the fit it is also useful to plot fit statistic profiles for specific parameters.
# For this we use `~gammapy.modeling.Fit.stat_profile()`.

# In[ ]:


profile = analysis.fit.stat_profile(parameter="lon_0")


# For a good fit and error estimate the profile should be parabolic, if we plot it:

# In[ ]:


total_stat = analysis.fit_result.total_stat
plt.plot(profile["values"], profile["stat"] - total_stat)
plt.xlabel("Lon (deg)")
plt.ylabel("Delta TS")


# ### Flux points

# In[ ]:


analysis.get_flux_points(source="crab")


# In[ ]:


plt.figure(figsize=(8, 5))
ax_sed, ax_residuals = analysis.flux_points.peek()
crab_spectrum = create_crab_spectral_model("hess_pl")
crab_spectrum.plot(
    ax=ax_sed,
    energy_range=[1, 10] * u.TeV,
    energy_power=2,
    flux_unit="erg-1 cm-2 s-1",
)


# ## Exercises
# 
# - Run a spectral analysis using reflected regions without stacking the datasets. You can use `AnalysisConfig.from_template("1d")` to get an example configuration file. Add the resulting flux points to the SED plotted above. 
# 
