
# coding: utf-8

# # 3D analysis
# 
# This tutorial shows how to run a 3D map-based analysis (two spatial and one energy axis).
# 
# The example data is three observations of the Galactic center region with CTA.

# ## Imports and versions

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.extern.pathlib import Path
from gammapy.data import DataStore
from gammapy.irf import EnergyDispersion
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.cube import MapMaker, PSFKernel, MapFit
from gammapy.cube.models import SkyModel
from gammapy.spectrum.models import PowerLaw
from gammapy.image.models import SkyGaussian, SkyPointSource
from regions import CircleSkyRegion


# In[3]:


get_ipython().system('gammapy info --no-envvar --no-dependencies --no-system')


# ## Make maps

# In[4]:


# Define which data to use
data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/")
obs_ids = [110380, 111140, 111159]
obs_list = data_store.obs_list(obs_ids)


# In[5]:


# Define map geometry (spatial and energy binning)
axis = MapAxis.from_edges(
    np.logspace(-1., 1., 10), unit="TeV", name="energy", interp="log"
)
geom = WcsGeom.create(
    skydir=(0, 0),
    binsz=0.02,
    width=(10, 8),
    coordsys="GAL",
    proj="CAR",
    axes=[axis],
)


# In[6]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, 4. * u.deg)\nmaps = maker.run(obs_list)')


# In[7]:


get_ipython().run_cell_magic('time', '', '# Make 2D images (for plotting, analysis will be 3D)\nimages = maker.make_images()')


# In[8]:


images["counts"].smooth(radius=0.2 * u.deg).plot(stretch="sqrt");


# In[9]:


images["background"].plot(stretch="sqrt");


# In[10]:


residual = images["counts"].copy()
residual.data -= images["background"].data
residual.smooth(5).plot(stretch="sqrt");


# ## Compute PSF kernel
# 
# For the moment we rely on the ObservationList.make_mean_psf() method.

# In[11]:


get_ipython().run_cell_magic('time', '', 'obs_list = data_store.obs_list(obs_ids)\nsrc_pos = SkyCoord(0, 0, unit="deg", frame="galactic")\n\ntable_psf = obs_list.make_mean_psf(src_pos)\npsf_kernel = PSFKernel.from_table_psf(\n    table_psf, maps["exposure"].geom, max_radius="0.3 deg"\n)')


# ## Compute energy dispersion

# In[12]:


get_ipython().run_cell_magic('time', '', 'energy_axis = geom.get_axis_by_name("energy")\nenergy = energy_axis.edges * energy_axis.unit\nedisp = obs_list.make_mean_edisp(\n    position=src_pos, e_true=energy, e_reco=energy\n)')


# ## Save maps
# 
# It's common to run the "precompute" step and the "likelihood fit" step separately,
# because often the "precompute" of maps, PSF and EDISP is slow if you have a lot of data.
# 
# Here it woudn't really be necessary, because the precompute step (everything above this cell)
# takes less than a minute.
# 
# But usually you would do it like this: write precomputed things to FITS files,
# and then read them from your script that does the likelihood fitting without
# having to run the precomputations again.

# In[13]:


# Write
path = Path("analysis_3d")
path.mkdir(exist_ok=True)
maps["counts"].write(str(path / "counts.fits"), overwrite=True)
maps["background"].write(str(path / "background.fits"), overwrite=True)
maps["exposure"].write(str(path / "exposure.fits"), overwrite=True)
psf_kernel.write(str(path / "psf.fits"), overwrite=True)
edisp.write(str(path / "edisp.fits"), overwrite=True)


# In[14]:


# Read
maps = {
    "counts": Map.read(str(path / "counts.fits")),
    "background": Map.read(str(path / "background.fits")),
    "exposure": Map.read(str(path / "exposure.fits")),
}
psf_kernel = PSFKernel.read(str(path / "psf.fits"))
edisp = EnergyDispersion.read(str(path / "edisp.fits"))


# ## Cutout
# 
# Let's cut out only part of the map, so that we can have a faster fit

# In[15]:


cmaps = {
    name: m.cutout(SkyCoord(0, 0, unit="deg", frame="galactic"), 1.5 * u.deg)
    for name, m in maps.items()
}
cmaps["counts"].sum_over_axes().plot(stretch="sqrt");


# ## Fit mask
# 
# To select a certain region and/or energy range for the fit we can create a fit mask.

# In[16]:


mask = Map.from_geom(cmaps["counts"].geom)

region = CircleSkyRegion(center=src_pos, radius=0.6 * u.deg)
mask.data = mask.geom.region_mask([region])

mask.get_image_by_idx((0,)).plot();


# In addition we also exclude the range below 0.3 TeV for the fit:

# In[17]:


coords = mask.geom.get_coord()
mask.data &= coords["energy"] > 0.3


# ## Model fit
# 
# - TODO: Add diffuse emission model? (it's 800 MB, maybe prepare a cutout)
# - TODO: compare against true model known for DC1

# In[18]:


spatial_model = SkyPointSource(lon_0="0.01 deg", lat_0="0.01 deg")
spectral_model = PowerLaw(
    index=2.2, amplitude="3e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


# In[19]:


get_ipython().run_cell_magic('time', '', 'fit = MapFit(\n    model=model,\n    counts=cmaps["counts"],\n    exposure=cmaps["exposure"],\n    background=cmaps["background"],\n    mask=mask,\n    psf=psf_kernel,\n    edisp=edisp,\n)\n\nfit.fit(opts_minuit={\'print_level\':1})')


# ## Check model fit
# 
# - plot counts spectrum for some on region (e.g. the one used in 1D spec analysis, 0.2 deg)
# - plot residual image for some energy band (e.g. the total one used here)

# In[20]:


# Parameter error are not synched back to
# sub model components automatically yet
spec = model.spectral_model.copy()
print(spec)


# In[21]:


# For now, we can copy the parameter error manually
spec.parameters.set_parameter_errors(
    {
        "index": model.parameters.error("index"),
        "amplitude": model.parameters.error("amplitude"),
    }
)
print(spec)


# In[22]:


energy_range = [0.1, 10] * u.TeV
spec.plot(energy_range, energy_power=2)
spec.plot_error(energy_range, energy_power=2);


# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1
# * Run the model fit with energy dispersion (pass edisp to MapFit)
