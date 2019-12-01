#!/usr/bin/env python
# coding: utf-8

# # First analysis
# 
# This notebook shows a simple example of a Crab analysis using the H.E.S.S. DL3 data release 1. It reduces the data to cube datasets and performs a simple 3D model fitting of the Crab nebula.
# 
# It uses the high level `Analysis` class to orchestrate data reduction. In its current state, `Analysis` supports the standard analysis cases of joint or stacked 3D and 1D analyses. It is instantiated with an `AnalysisConfig` object that gives access to analysis parameters either directly or via a YAML config file. 
# 
# To see what is happening under-the-hood and to get an idea of the internal API, a second notebook performs the same analysis without using the `Analysis` class. 
# 
# We will first show how to configure and run a stacked 3D analysis. The structure of the tutorial follows a typical analysis:
# 
# - Analysis configuration
# - Observation selection
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


from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model


# ## Analysis configuration
# 
# For configuration of the analysis we use the [YAML](https://en.wikipedia.org/wiki/YAML) data format. YAML is a machine readable serialisation format, that is also friendly for humans to read. In this tutorial we will write the configuration file just using Python strings, but of course the file can be created and modified with any text editor of your choice.
# 
# Here is what the configuration for our analysis looks like:

# In[ ]:


config = AnalysisConfig()
# the AnalysisConfig gives access to the various parameters used from logging to reduced dataset geometries
print(config)


# ### Setting the data to use

# We want to use Crab runs from the H.E.S.S. DL3-DR1. We define here the datastore and a cone search of observations pointing with 5 degrees of the Crab nebula.
# Parameters can be set directly or as a python dict.

# In[ ]:


# We define the datastore containing the data
config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1"

# We define the cone search parameters
config.observations.obs_cone.frame = "icrs"
config.observations.obs_cone.lon = "83.633 deg"
config.observations.obs_cone.lat = "22.014 deg"
config.observations.obs_cone.radius = "5 deg"

# Equivalently we could have set parameters with a python dict
# config.observations.obs_cone = {"frame": "icrs", "lon": "83.633 deg", "lat": "22.014 deg", "radius": "5 deg"}


# ### Setting the reduced datasets geometry

# In[ ]:


# We want to perform a 3D analysis
config.datasets.type = "3d"
# We want to stack the data into a single reduced dataset
config.datasets.stack = True

# We fiw the WCS geometry of the datasets
config.datasets.geom.wcs.skydir = {
    "lon": "83.633 deg",
    "lat": "22.014 deg",
    "frame": "icrs",
}
config.datasets.geom.wcs.fov = {"width": "2 deg", "height": "2 deg"}
config.datasets.geom.wcs.binsize = "0.02 deg"

# We now fix the energy axis for the counts map
config.datasets.geom.axes.energy.min = "1 TeV"
config.datasets.geom.axes.energy.max = "10 TeV"
config.datasets.geom.axes.energy.nbins = 4

# We now fix the energy axis for the IRF maps (exposure, etc)
config.datasets.geom.axes.energy_true.min = "0.5 TeV"
config.datasets.geom.axes.energy_true.max = "20 TeV"
config.datasets.geom.axes.energy.nbins = 10


# ### Setting modeling and fitting parameters
# `Analysis` can perform a few modeling and fitting tasks besides data reduction. Parameters have then to be passed to the configuration object.

# In[ ]:


config.fit.fit_range.min = 1 * u.TeV
config.fit.fit_range.max = 10 * u.TeV
config.flux_points.energy = {"min": "1 TeV", "max": "10 TeV", "nbins": 3}


# We're all set. 
# But before we go on let's see how to save or import `AnalysisConfig` objects though YAML files.

# ### Using YAML configuration files
# 
# One can export/import the `AnalysisConfig` to/from a YAML file.

# In[ ]:


config.write("config.yaml", overwrite=True)


# In[ ]:


config = AnalysisConfig.read("config.yaml")
print(config)


# ## Running the analysis
# 
# We first create an `~gammapy.analysis.Analysis` object from our configuration.

# In[ ]:


analysis = Analysis(config)


# ###  Observation selection
# 
# We can directly select and load the observations from disk using `~gammapy.analysis.Analysis.get_observations()`:

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


analysis.set_models(model_config)


# In[ ]:


print(analysis.models)


# In[ ]:


print(analysis.models["crab"])


# Finally we run the fit:

# In[ ]:


analysis.run_fit()


# In[ ]:


print(analysis.fit_result)


# This is how we can write the model back to file again:

# In[ ]:


analysis.models.to_yaml("model-best-fit.yaml")


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
