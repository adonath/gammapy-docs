#!/usr/bin/env python
# coding: utf-8

# # Joint 3D Analysis
# 
# In this tutorial we show how to run a joint 3D map-based analysis using three example observations of the Galactic center region with CTA. We start with the required imports:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion


# In[ ]:


from gammapy.data import DataStore
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.cube import MapDatasetMaker, MapDataset
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.modeling import Fit


# ## Prepare modeling input data
# 
# We first use the `~gammapy.data.DataStore` object to access the CTA observations and retrieve a list of observations by passing the observations IDs to the `~gammapy.data.DataStore.get_observations()` method:

# In[ ]:


# Define which data to use and print some information
data_store = DataStore.from_dir("$GAMMAPY_DATA/cta-1dc/index/gps/")


# In[ ]:


# Select some observations from these dataset by hand
obs_ids = [110380, 111140, 111159]
observations = data_store.get_observations(obs_ids)


# ### Prepare datasets
# 
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


# In addition we define the center coordinate and the FoV offset cut:

# In[ ]:


# Source position
src_pos = SkyCoord(0, 0, unit="deg", frame="galactic")

# FoV max
offset_max = 4 * u.deg


# The datasets are prepared by using the `~gammapy.cube.MapDatasetMaker.run()` method and passing the `observation`.

# In[ ]:


path = Path("analysis_3d_joint")
path.mkdir(exist_ok=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'maker = MapDatasetMaker(geom=geom, offset_max=offset_max)\nfor obs in observations:\n    dataset = maker.run(obs)\n\n    # TODO: remove once IRF maps are handled correctly in fit\n    dataset.edisp = dataset.edisp.get_energy_dispersion(\n        position=src_pos, e_reco=energy_axis.edges\n    )\n    dataset.psf = dataset.psf.get_psf_kernel(\n        position=src_pos, geom=geom, max_radius="0.3 deg"\n    )\n    dataset.write(\n        f"analysis_3d_joint/dataset-obs-{obs.obs_id}.fits", overwrite=True\n    )')


# ## Likelihood fit
# 
# ### Defining model and reading datasets
# As first step we define a source model:

# In[ ]:


spatial_model = PointSpatialModel(
    lon_0="-0.05 deg", lat_0="-0.05 deg", frame="galactic"
)
spectral_model = PowerLawSpectralModel(
    index=2.4, amplitude="2.7e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


# Now we read the maps and IRFs and create the dataset for each observation:

# In[ ]:


datasets = []

for obs_id in obs_ids:
    dataset = MapDataset.read(f"analysis_3d_joint/dataset-obs-{obs_id}.fits")
    dataset.model = model
    dataset.background_model.tilt.frozen = False

    # optionally define a safe energy threshold
    emin = None
    dataset.mask_safe = dataset.counts.geom.energy_mask(emin=emin)
    datasets.append(dataset)


# In[ ]:


fit = Fit(datasets)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'result = fit.run()')


# In[ ]:


print(result)


# Best fit parameters:

# In[ ]:


fit.datasets.parameters.to_table()


# The information which parameter belongs to which dataset is not listed explicitly in the table (yet), but the order of parameters is conserved. You can always access the underlying object tree as well to get specific parameter values:

# In[ ]:


for dataset in datasets:
    print(dataset.background_model.norm.value)


# ## Plotting residuals

# Each `~gammapy.cube.MapDataset` object is equipped with a method called `~gammapy.cube.MapDataset.plot_residuals()`, which displays the spatial and spectral residuals (computed as *counts-model*) for the dataset. Optionally, these can be normalized as *(counts-model)/model* or *(counts-model)/sqrt(model)*, by passing the parameter `norm='model` or `norm=sqrt_model`.
# 
# First of all, let's define a region for the spectral extraction:

# In[ ]:


region = CircleSkyRegion(spatial_model.position, radius=0.15 * u.deg)


# We can now inspect the residuals for each dataset, separately:

# In[ ]:


ax_image, ax_spec = datasets[0].plot_residuals(
    region=region, vmin=-0.5, vmax=0.5, method="diff"
)


# In[ ]:


datasets[1].plot_residuals(region=region, vmin=-0.5, vmax=0.5);


# In[ ]:


datasets[2].plot_residuals(region=region, vmin=-0.5, vmax=0.5);


# Finally, we can compute a stacked dataset:

# In[ ]:


residuals_stacked = Map.from_geom(geom)

for dataset in datasets:
    residuals = dataset.residuals()
    coords = residuals.geom.get_coord()

    residuals_stacked.fill_by_coord(coords, residuals.data)


# In[ ]:


residuals_stacked.sum_over_axes().smooth("0.1 deg").plot(
    vmin=-1, vmax=1, cmap="coolwarm", add_cbar=True, stretch="linear"
);

