
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
from gammapy.image.models import SkyGaussian


# In[3]:


get_ipython().system('gammapy info --no-envvar --no-dependencies --no-system')


# ## Make maps

# In[4]:


# Define which data to use
data_store = DataStore.from_dir(
    '$GAMMAPY_EXTRA/datasets/cta-1dc/index/gps/'
)
obs_ids = [110380, 111140, 111159]
# obs_ids = [110380]
obs_list = data_store.obs_list(obs_ids)


# In[5]:


# Define map geometry (spatial and energy binning)
axis = MapAxis.from_edges(
    np.logspace(-1., 1., 10), unit='TeV', name='energy'
)
geom = WcsGeom.create(
    skydir=(0, 0), binsz=0.02, width=(20, 15),
    coordsys='GAL', proj='CAR',
    axes=[axis],
)


# In[6]:


get_ipython().run_cell_magic('time', '', 'maker = MapMaker(geom, 4. * u.deg)\nmaps = maker.run(obs_list)')


# In[7]:


maps['counts'].sum_over_axes().smooth(radius=0.2*u.deg).plot(stretch='sqrt');


# In[8]:


maps['background'].sum_over_axes().plot(stretch='sqrt');


# In[9]:


residual = maps['counts'].copy()
residual.data -= maps['background'].data
residual.sum_over_axes().smooth(5).plot(stretch='sqrt');


# ## Compute PSF kernel
# 
# For the moment we rely on the ObservationList.make_mean_psf() method.

# In[10]:


obs_list = data_store.obs_list(obs_ids)
src_pos = SkyCoord(0, 0, unit='deg', frame='galactic')

table_psf = obs_list.make_mean_psf(src_pos)
psf_kernel = PSFKernel.from_table_psf(
    table_psf,
    maps['exposure'].geom,
    max_radius='0.3 deg',
)


# ## Compute energy dispersion

# In[11]:


energy_axis = geom.get_axis_by_name('energy')
energy = energy_axis.edges * energy_axis.unit
edisp = obs_list.make_mean_edisp(position=src_pos, e_true=energy, e_reco=energy)


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

# In[12]:


# Write
path = Path('analysis_3d')
path.mkdir(exist_ok=True)
maps['counts'].write(str(path / 'counts.fits'), overwrite=True)
maps['background'].write(str(path / 'background.fits'), overwrite=True)
maps['exposure'].write(str(path / 'exposure.fits'), overwrite=True)
psf_kernel.write(str(path / 'psf.fits'), overwrite=True)
edisp.write(str(path / 'edisp.fits'), overwrite=True)


# In[13]:


# Read
maps = {
    'counts': Map.read(str(path / 'counts.fits')),
    'background': Map.read(str(path / 'background.fits')),
    'exposure': Map.read(str(path / 'exposure.fits')),
}
psf_kernel = PSFKernel.read(str(path / 'psf.fits'))
edisp = EnergyDispersion.read(str(path / 'edisp.fits'))


# ## Cutout
# 
# Let's cut out only part of the map, so that we can have a faster fit

# In[14]:


cmaps = {
    name: m.cutout(SkyCoord(0, 0, unit='deg', frame='galactic'), 1.5 * u.deg)
    for name, m in maps.items()
}
cmaps['counts'].sum_over_axes().plot(stretch='sqrt');


# ## Model fit
# 
# - TODO: Add diffuse emission model? (it's 800 MB, maybe prepare a cutout)
# - TODO: make it faster (less than 1 minute)
# - TODO: compare against true model known for DC1

# In[15]:


spatial_model = SkyGaussian(
    lon_0='0 deg',
    lat_0='0 deg',
    sigma='0.1 deg',
)
spectral_model = PowerLaw(
    index=2.2,
    amplitude='3e-12 cm-2 s-1 TeV-1',
    reference='1 TeV',
)
model = SkyModel(
    spatial_model=spatial_model,
    spectral_model=spectral_model,
)


# In[16]:


# For now, users have to set initial step sizes
# to help MINUIT to converge
model.parameters.set_parameter_errors({
    'lon_0': '0.01 deg',
    'lat_0': '0.01 deg',
    'sigma': '0.1 deg',
    'index': 0.1,
    'amplitude': '1e-13 cm-2 s-1 TeV-1',
})

# model.parameters['lon_0'].frozen = True
# model.parameters['lat_0'].frozen = True
# model.parameters['sigma'].frozen = True
# model.parameters['sigma'].min = 0


# In[17]:


get_ipython().run_cell_magic('time', '', "fit = MapFit(\n    model=model,\n    counts=cmaps['counts'],\n    exposure=cmaps['exposure'],\n    background=cmaps['background'],\n    psf=psf_kernel,\n#     edisp=edisp,\n)\n\nfit.fit()")


# In[18]:


print(model.parameters)


# ## Check model fit
# 
# - plot counts spectrum for some on region (e.g. the one used in 1D spec analysis, 0.2 deg)
# - plot residual image for some energy band (e.g. the total one used here)

# In[19]:


# Parameter error are not synched back to
# sub model components automatically yet
spec = model.spectral_model.copy()
print(spec)


# In[20]:


# For now, we can copy the parameter error manually
spec.parameters.set_parameter_errors({
    'index': model.parameters.error('index'),
    'amplitude': model.parameters.error('amplitude'),
})
print(spec)


# In[21]:


energy_range = [0.1, 10] * u.TeV
spec.plot(energy_range, energy_power=2)
spec.plot_error(energy_range, energy_power=2)


# ## Exercises
# 
# * Analyse the second source in the field of view: G0.9+0.1
# * Run the model fit with energy dispersion (pass edisp to MapFit)
