
# coding: utf-8

# # Source detection with Gammapy
# 
# ## Introduction
# 
# This notebook show how to do source detection with Gammapy using one of the methods available in [gammapy.detect](http://docs.gammapy.org/dev/detect/index.html).
# 
# We will do this:
# 
# * produce 2-dimensional test-statistics (TS) images using Fermi-LAT 2FHL high-energy Galactic plane survey dataset
# * run a peak finder to make a source catalog
# * do some simple measurements on each source
# * compare to the 2FHL catalog
# 
# Note that what we do here is a quick-look analysis, the production of real source catalogs use more elaborate procedures.
# 
# We will work with the following functions and classes:
# 
# * [photutils](http://photutils.readthedocs.io/en/latest/) and specifically the [photutils.detection.find_peaks](http://photutils.readthedocs.io/en/latest/api/photutils.detection.find_peaks.html) function.
# * [gammapy.maps.WcsNDMap](http://docs.gammapy.org/dev/api/gammapy.maps.WcsNDMap.html)
# * [gammapy.detect.TSMapEstimator](http://docs.gammapy.org/dev/api/gammapy.detect.TSMapEstimator.html)

# ## Setup
# 
# As always, let's get started with some setup ...

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord
from photutils.detection import find_peaks
from gammapy.maps import Map
from gammapy.detect import TSMapEstimator
from gammapy.catalog import source_catalogs


# ## Compute TS image

# In[ ]:


# Load data from files
filename = '../datasets/fermi_survey/all.fits.gz'
maps = {
    'counts': Map.read(filename, hdu='COUNTS'),
    'background': Map.read(filename, hdu='BACKGROUND'),
    'exposure': Map.read(filename, hdu='EXPOSURE'),
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Compute a source kernel (source template) in oversample mode,\n# PSF is not taken into account\nkernel = Gaussian2DKernel(2.5, mode='oversample')\nestimator = TSMapEstimator()\nimages = estimator.run(maps, kernel)")


# ## Plot images

# In[ ]:


plt.figure(figsize=(18, 4))
images['sqrt_ts'].plot(vmin=0, vmax=10);


# In[ ]:


plt.figure(figsize=(18, 4))
images['flux'].plot(vmin=0, vmax=1e-9, stretch='sqrt');


# In[ ]:


plt.figure(figsize=(18, 4))
images['niter'].plot(vmin=0, vmax=20);


# ## Source catalog
# 
# Let's run a peak finder on the `sqrt_ts` image to get a list of sources (positions and peak `sqrt_ts` values).

# In[ ]:


sources = find_peaks(
    data=np.nan_to_num(images['sqrt_ts'].data),
    threshold=8,
    wcs=images['sqrt_ts'].geom.wcs,
)
sources


# In[ ]:


# Plot sources on top of significance sky image
images['sqrt_ts'].cutout(
    position=SkyCoord(0, 0, unit='deg', frame='galactic'),
    width=(8*u.deg, 20*u.deg), mode='trim',
).plot()

plt.gca().scatter(
    sources['icrs_ra_peak'], sources['icrs_dec_peak'],
    transform=plt.gca().get_transform('icrs'),
    color='none', edgecolor='white', marker='o', s=600, lw=1.5,
);


# ## Measurements
# 
# * TODO: show cutout for a few sources and some aperture photometry measurements (e.g. energy distribution, significance, flux)

# In[ ]:


# TODO


# ## Compare to 2FHL
# 
# TODO

# In[ ]:


fermi_2fhl = source_catalogs['2fhl']
fermi_2fhl.table[:5][['Source_Name', 'GLON', 'GLAT']]


# ## Exercises
# 
# TODO: put one or more exercises

# In[ ]:


# Start exercises here!


# ## What next?
# 
# In this notebook, we have seen how to work with images and compute TS images from counts data, if a background estimate is already available.
# 
# Here's some suggestions what to do next:
# 
# - TODO: point to background estimation examples
# - TODO: point to other docs ...
