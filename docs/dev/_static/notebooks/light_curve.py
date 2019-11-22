#!/usr/bin/env python
# coding: utf-8

# # Light curve estimation
# 
# ## Introduction
# 
# This tutorial presents a new light curve estimator that works with dataset objects. We will demonstrate how to compute a `~gammapy.time.LightCurve` from 3D data cubes as well as 1D spectral data using the `~gammapy.cube.MapDataset`, `~gammapy.spectrum.SpectrumDatasetOnOff` and `~gammapy.time.LightCurveEstimator` classes. 
# 
# We will compute two LCs: one per observation and one by night for which you have to provide the time intervals
#     
# We will use the four Crab nebula observations from the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/) and compute per-observation fluxes. The Crab nebula is not known to be variable at TeV energies, so we expect constant brightness within statistical and systematic errors.
# 
# ## Setup
# 
# As usual, we'll start with some general imports...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
import logging

from astropy.time import Time

log = logging.getLogger(__name__)


# Now let's import gammapy specific classes and functions

# In[ ]:


from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.modeling.models import PointSpatialModel
from gammapy.modeling.models import SkyModel
from gammapy.cube import MapDatasetMaker, MapDataset, SafeMaskMaker
from gammapy.maps import WcsGeom, MapAxis
from gammapy.time import LightCurveEstimator
from gammapy.analysis import Analysis, AnalysisConfig 


# ## Analysis configuration 
# For the 1D and 3D extraction, we will use the same CrabNebula configuration than in the notebook analysis_1.ipynb using the high level interface of Gammapy.
# 
# From the high level interface, the datareduction for those observations is performed as followed

# ### 3D data reduction + Fit
# 

# #### Data reduction

# In[ ]:


conf_3d=AnalysisConfig.from_template("3d") 
# We want to extract the data by observation and therefore to not stack them
conf_3d.settings['datasets']['stack-datasets']=False
#Fixing more physical binning
conf_3d.settings['datasets']['geom']['axes'][0]['lo_bnd']=0.7
conf_3d.settings['datasets']['geom']['axes'][0]['nbin']=10
conf_3d.settings['datasets']['energy-axis-true']['lo_bnd']=0.1
conf_3d.settings['datasets']['energy-axis-true']['hi_bnd']=20
conf_3d.settings['datasets']['energy-axis-true']['nbin']=20
conf_3d.settings['datasets']['geom']['width']=[2,2]
conf_3d.settings['datasets']['geom']['binsz']=0.02

ana_3d=Analysis(conf_3d)
ana_3d.get_observations()
ana_3d.get_datasets()


# ##### 3D Fit

# Define the model to be fitted

# In[ ]:


target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")
spatial_model = PointSpatialModel(
   lon_0=target_position.ra, lat_0=target_position.dec, frame="icrs"
)
spectral_model = PowerLawSpectralModel(
   index=2.6,
   amplitude=2.0e-11 * u.Unit("1 / (cm2 s TeV)"),
   reference=1 * u.TeV,
)
spectral_model.parameters["index"].frozen = False
sky_model = SkyModel(
   spatial_model=spatial_model, spectral_model=spectral_model, name="crab"
)
sky_model.parameters["lon_0"].frozen = True
sky_model.parameters["lat_0"].frozen = True


# We assign them the model to be fitted to each dataset

# In[ ]:


model = {}
model["components"] = [sky_model.to_dict()]
ana_3d.set_model(model=model)


# Do the fit

# In[ ]:


ana_3d.run_fit()


# ### 1D data reduction

# #### Data reduction

# In[ ]:


conf_1d=AnalysisConfig.from_template("1d") 
# We want to extract the data by observation and therefore to not stack them
conf_1d.settings['datasets']['stack-datasets']=False
conf_1d.settings['datasets']['containment_correction']=True
conf_1d.settings['datasets']['geom']['axes'][0]['lo_bnd']=0.7
conf_1d.settings['datasets']['geom']['axes'][0]['hi_bnd']=40
conf_1d.settings['datasets']['geom']['axes'][0]['nbin']=40

ana_1d=Analysis(conf_1d)     
ana_1d.get_observations() 
ana_1d.get_datasets() 


# #### 1D Fit

# We assign the spectral model to be fitted to each dataset

# In[ ]:


model = {}
model["components"] = [sky_model.to_dict()]
ana_1d.set_model(model=model)


# Do the fit

# In[ ]:


ana_1d.run_fit()


# ## Light Curve estimation: by observation
# We can now create the light curve estimator by passing it the list of datasets. We can optionally ask for parameters reoptimization during fit, e.g. to fit background normalization in each time bin.
# 
# By default, the LightCurveEstimator is performed by dataset, here one dataset=one observation

# ### 3d

# In[ ]:


lc_maker_3d = LightCurveEstimator(ana_3d.datasets, source="crab", reoptimize=True)
lc_3d = lc_maker_3d.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# The LightCurve object contains a table which we can explore.

# In[ ]:


lc_3d.table["time_min", "time_max", "flux", "flux_err"]


# ### 1d

# If you want to add a fit range for each of you time intervals when computing the LC.

# In[ ]:


e_min_fit = 0.8 * u.TeV
e_max_fit = 10 * u.TeV
for dataset in ana_1d.datasets:
    mask_fit = dataset.counts.energy_mask(emin=e_min_fit, emax=e_max_fit)
    dataset.mask_fit = mask_fit


# In[ ]:


lc_maker_1d = LightCurveEstimator(ana_1d.datasets, source="crab", reoptimize=False)
lc_1d = lc_maker_1d.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# ### Compare results
# 
# Finally we compare the result for the 1D and 3D lightcurve in a single figure:

# In[ ]:


ax = lc_1d.plot(marker="o", label="1D")
lc_3d.plot(ax=ax, marker="o", label="3D")
plt.legend()


# ## LC estimation by night
# We define the time intervals to compute the LC by night, here three nights.

# In[ ]:


time_intervals = [Time([53343.5,53344.5], format='mjd', scale='utc'),
                Time([53345.5,53346.5], format='mjd', scale='utc'),
                Time([53347.5,53348.5], format='mjd', scale='utc')
                 ]


# Compute 1D LC

# In[ ]:


lc_maker_1d_bynight = LightCurveEstimator(ana_1d.datasets, time_intervals=time_intervals,source="crab", reoptimize=False)
lc_1d_bynight = lc_maker_1d_bynight.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# Compute 3D LC

# In[ ]:


lc_maker_3d_bynight = LightCurveEstimator(ana_3d.datasets, time_intervals=time_intervals, source="crab", reoptimize=True)
lc_3d_bynight = lc_maker_3d_bynight.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# Compare LC by night

# ax = lc_1d_bynight.plot(marker="o", label="1D")
# lc_3d_bynight.plot(ax=ax, marker="o", label="3D")
# plt.legend()

# In[ ]:


ax = lc_1d_bynight.plot(marker="o", label="1D")
lc_3d_bynight.plot(ax=ax, marker="o", label="3D")
plt.legend()


# In[ ]:





# In[ ]:




