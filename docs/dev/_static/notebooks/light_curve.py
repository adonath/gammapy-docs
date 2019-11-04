#!/usr/bin/env python
# coding: utf-8

# # Light curve estimation
# 
# ## Introduction
# 
# This tutorial presents a new light curve estimator that works with dataset objects. We will demonstrate how to compute a `~gammapy.time.LightCurve` from 3D data cubes as well as 1D spectral data using the `~gammapy.cube.MapDataset`, `~gammapy.spectrum.SpectrumDatasetOnOff` and `~gammapy.time.LightCurveEstimator` classes. 
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

log = logging.getLogger(__name__)


# Now let's import gammapy specific classes and functions

# In[ ]:


from gammapy.data import DataStore
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.modeling.models import PointSpatialModel
from gammapy.modeling.models import SkyModel
from gammapy.cube import MapDatasetMaker, MapDataset
from gammapy.maps import WcsGeom, MapAxis
from gammapy.time import LightCurveEstimator


# ## Select the data
# 
# We look for relevant observations in the datastore.

# In[ ]:


data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
mask = data_store.obs_table["TARGET_NAME"] == "Crab"
obs_ids = data_store.obs_table["OBS_ID"][mask].data
crab_obs = data_store.get_observations(obs_ids)


# ## Define time intervals
# We create a list of time intervals. Here we use one time bin per observation.

# In[ ]:


time_intervals = [(obs.tstart, obs.tstop) for obs in crab_obs]


# ## 3D data reduction 
# 
# ### Define the analysis geometry
# 
# Here we define the geometry used in the analysis. We use the same WCS map structure but we use two different binnings for reco and true energy axes. This allows for a broader coverage of the response.

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
    0.1, 20, 20, unit="TeV", name="energy", interp="log"
)

offset_max = 2 * u.deg


# ### Define the 3D model 
# 
# The light curve is based on a 3D fit of a map dataset in time bins. We therefore need to define the source model to be applied. Here a point source with power law spectrum. We freeze its parameters assuming they were previously extracted

# In[ ]:


# Define the source model - Use a pointsource + integrated power law model to directly get flux

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
    spatial_model=spatial_model, spectral_model=spectral_model, name=""
)
sky_model.parameters["lon_0"].frozen = True
sky_model.parameters["lat_0"].frozen = True


# ### Make the map datasets
# 
# The following function is in charge of the MapDataset production. It will later be fully covered in the data reduction chain 

# Now we perform the actual data reduction in time bins

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndatasets = []\n\nmaker = MapDatasetMaker(\n    geom=geom, energy_axis_true=energy_axis_true, offset_max=offset_max\n)\n\nfor time_interval in time_intervals:\n    # get filtered observation lists in time interval\n    observations = crab_obs.select_time(time_interval)\n\n    # Proceed with further analysis only if there are observations\n    # in the selected time window\n    if len(observations) == 0:\n        log.warning(f"No observations in time interval: {time_interval}")\n        continue\n\n    stacked = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true)\n\n    for obs in observations:\n        dataset = maker.run(obs)\n        stacked.stack(dataset)\n\n    # TODO: remove once IRF maps are handled correctly in fit\n    stacked.edisp = stacked.edisp.get_energy_dispersion(\n        position=target_position, e_reco=energy_axis.edges\n    )\n\n    stacked.psf = stacked.psf.get_psf_kernel(\n        position=target_position,\n        geom=stacked.exposure.geom,\n        max_radius="0.3 deg",\n    )\n\n    stacked.counts.meta["t_start"] = time_interval[0]\n    stacked.counts.meta["t_stop"] = time_interval[1]\n    datasets.append(stacked)')


# ## Light Curve estimation: the 3D case
# 
# Now that we have created the datasets we assign them the model to be fitted:

# In[ ]:


for dataset in datasets:
    # Copy the source model
    model = sky_model.copy(name="crab")
    dataset.model = model


# We can now create the light curve estimator by passing it the list of datasets. 
# We can optionally ask for parameters reoptimization during fit, e.g. to fit background normalization in each time bin.

# In[ ]:


lc_maker = LightCurveEstimator(datasets, source="crab", reoptimize=True)


# We now run the estimator once we pass it the energy interval on which to compute the integral flux of the source.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'lc = lc_maker.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)')


# The LightCurve object contains a table which we can explore.

# In[ ]:


lc.table["time_min", "time_max", "flux", "flux_err"]


# We finally plot the light curve

# In[ ]:


lc.plot(marker="o")


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
    SafeMaskMaker,
)


# ### Defining the geometry
# 
# We need to define the ON extraction region. We will keep the same reco and true energy axes as in 3D.

# In[ ]:


# Target definition
e_reco = np.logspace(-1, np.log10(40), 40) * u.TeV
e_true = np.logspace(np.log10(0.05), 2, 100) * u.TeV

on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# In[ ]:


dataset_maker = SpectrumDatasetMaker(
    region=on_region, e_reco=e_reco, e_true=e_true, containment_correction=True
)
bkg_maker = ReflectedRegionsBackgroundMaker()
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)


# ### Creation of the datasets

# In[ ]:


datasets_1d = []

for time_interval in time_intervals:
    observation = crab_obs.select_time(time_interval)[0]

    dataset = dataset_maker.run(
        observation, selection=["counts", "aeff", "edisp"]
    )

    dataset.counts.meta = dict()
    dataset.counts.meta["t_start"] = time_interval[0]
    dataset.counts.meta["t_stop"] = time_interval[1]

    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets_1d.append(dataset_on_off)


# ## Light Curve estimation for 1D spectra
# 
# Now that we've reduced the 1D data we assign again the model to the datasets 

# In[ ]:


for dataset in datasets_1d:
    # Copy the source model
    model = spectral_model.copy()
    model.name = "crab"
    dataset.model = model


# We can now call the LightCurveEstimator in a perfectly identical manner.

# In[ ]:


lc_maker_1d = LightCurveEstimator(datasets_1d, source="crab", reoptimize=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lc_1d = lc_maker_1d.run(e_ref=1 * u.TeV, e_min=1.0 * u.TeV, e_max=10.0 * u.TeV)')


# ### Compare results
# 
# Finally we compare the result for the 1D and 3D lightcurve in a single figure:

# In[ ]:


ax = lc_1d.plot(marker="o", label="1D")
lc.plot(ax=ax, marker="o", label="3D")
plt.legend()

