#!/usr/bin/env python
# coding: utf-8

# # Light curve- Flare
# 
# To see the general presentation on our light curve estimator, please refer to the notebook `light_curve.ipynb`
# 
# Here we present the way to compute a light curve on smaller time interval than the duration of an observation.
# 
# We will use the Crab nebula observations from the H.E.S.S. first public test data release 
# 
# We define time intervals from the time start to the time stop of the observations, spaced on 15 minutes. To estimate the light curve in all of those time bins, we use the joint fits result from the light_curve.ipynb nobtebook extracted on the same dataset.
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
from astropy.time import Time
import logging

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
from gammapy.modeling import Fit


# ## Select the data
# 
# We look for relevant observations in the datastore.

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_ids = data_store.obs_table["OBS_ID"][mask].data
crab_obs = data_store.get_observations(obs_ids)


# ## Define time intervals
# We create a list of time intervals. We define time intervals from the time start to the time stop of the observations, spaced on 15 minutes. 

# In[ ]:


time_intervals = [
    ["2004-12-04T22:00", "2004-12-04T22:15"],
    ["2004-12-04T22:15", "2004-12-04T22:30"],
    ["2004-12-04T22:30", "2004-12-04T22:45"],
    ["2004-12-04T22:45", "2004-12-04T23:00"],
    ["2004-12-04T23:00", "2004-12-04T23:15"],
    ["2004-12-04T23:15", "2004-12-04T23:30"],
    ["2004-12-06T23:00", "2004-12-06T23:15"],
    ["2004-12-06T23:15", "2004-12-06T23:30"],
    ["2004-12-08T21:45", "2004-12-08T22:00"],
    ["2004-12-08T22:00", "2004-12-08T22:15"],
]
time_intervals = [Time(_) for _ in time_intervals]


# ## Get filtered observation lists in time intervals
# 
# Return a list of Observations, defined on the previous time_intervals

# In[ ]:


observations = crab_obs.select_time(time_intervals)


# ## 3D data reduction 

# ### Defining the geometry

# In[ ]:


# Target definition
target_position = SkyCoord(ra=83.63308, dec=22.01450, unit="deg")

# Define geoms
emin, emax = [0.7, 10] * u.TeV
energy_axis = MapAxis.from_bounds(
    emin.value, emax.value, 10, unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=target_position,
    binsz=0.02,
    width=(2, 2),
    coordsys="CEL",
    proj="CAR",
    axes=[energy_axis],
)

energy_axis_true = MapAxis.from_bounds(
    0.1, 50, 20, unit="TeV", name="energy", interp="log"
)

offset_max = 2 * u.deg


# ### Make the map datasets
# 
# The following function is in charge of the MapDataset production. It will later be fully covered in the data reduction chain
# 
# Now we perform the actual data reduction in the time_intervals.

# In[ ]:


datasets_3d = []
maker = MapDatasetMaker()
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)

reference = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)

position = SkyCoord(83.63308 * u.deg, 22.01450 * u.deg, frame="icrs")
for obs in observations:

    dataset = maker.run(reference, obs)
    dataset = maker_safe_mask.run(dataset, obs)

    # TODO: remove once IRF maps are handled correctly in fit
    dataset.edisp = dataset.edisp.get_energy_dispersion(
        position=position, e_reco=energy_axis.edges
    )

    dataset.psf = dataset.psf.get_psf_kernel(
        position=position, geom=dataset.exposure.geom, max_radius="0.3 deg"
    )
    datasets_3d.append(dataset)


# ### Define the 3D model 
# 
# The light curve is based on a 3D fit of a map dataset in time bins. We therefore need to define the source model to be applied. Here a point source with power law spectrum. We freeze its parameters since they are extracted from the fit in the notebook `light_curve.ipynb`

# In[ ]:


# Define the source model - Use a pointsource + integrated power law model to directly get flux
spatial_model = PointSpatialModel(
    lon_0=target_position.ra, lat_0=target_position.dec, frame="icrs"
)

spectral_model = PowerLawSpectralModel(
    index=2.587,
    amplitude=4.305e-11 * u.Unit("1 / (cm2 s TeV)"),
    reference=1 * u.TeV,
)
spectral_model.parameters["index"].frozen = False

sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="crab"
)
sky_model.parameters["lon_0"].frozen = True
sky_model.parameters["lat_0"].frozen = True


# We affect to each dataset, the spatial model 

# In[ ]:


for dataset in datasets_3d:
    dataset.model = sky_model


# We can now create the light curve estimator by passing it the list of datasets. 
# We can optionally ask for parameters reoptimization during fit, e.g. to fit background normalization in each time bin.
# 
# The LC will be compute by default on the datasets GTI.

# In[ ]:


lc_maker_3d = LightCurveEstimator(datasets_3d, source="crab", reoptimize=True)
lc_3d = lc_maker_3d.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# ## Performing the same analysis with 1D spectra
# 
# ### First the relevant imports
# 
# We import the missing classes for spectral data reduction

# In[ ]:


from regions import CircleSkyRegion
from astropy.coordinates import Angle
from gammapy.spectrum import (
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)


# ### Defining the geometry
# 
# We need to define the ON extraction region. We will keep the same reco and true energy axes as in 3D.

# In[ ]:


# Target definition
e_reco = MapAxis.from_energy_bounds(0.1, 40, 100, "TeV").edges
e_true = MapAxis.from_energy_bounds(0.05, 100, 100, "TeV").edges

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# In[ ]:


dataset_maker = SpectrumDatasetMaker(
    region=on_region, e_reco=e_reco, e_true=e_true, containment_correction=True
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


# ### Creation of the datasets
# 
# Now we perform the actual data reduction in the time_intervals.

# In[ ]:


datasets_1d = []

for obs in observations:
    dataset = dataset_maker.run(obs, selection=["counts", "aeff", "edisp"])

    dataset_on_off = bkg_maker.run(dataset, obs)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, obs)
    datasets_1d.append(dataset_on_off)


# ### Define Spectral Model

# In[ ]:


spectral_model_1d = PowerLawSpectralModel(
    index=2.702,
    amplitude=4.712e-11 * u.Unit("1 / (cm2 s TeV)"),
    reference=1 * u.TeV,
)
spectral_model_1d.parameters["index"].frozen = False


# We affect to each dataset it spectral model

# In[ ]:


for dataset in datasets_1d:
    dataset.model = sky_model


# We can now create the light curve estimator by passing it the list of datasets.
# 
# The LC will be compute by default on the datasets GTI.
# 

# In[ ]:


lc_maker_1d = LightCurveEstimator(datasets_1d, source="crab", reoptimize=True)


# In[ ]:


lc_1d = lc_maker_1d.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)


# ## Compare results between 1D and 3D LC
# 
# Finally we compare the result for the 1D and 3D lightcurve in a single figure:

# In[ ]:


ax = lc_1d.plot(marker="o", label="1D")
lc_3d.plot(ax=ax, marker="o", label="3D")
plt.legend()


# 
