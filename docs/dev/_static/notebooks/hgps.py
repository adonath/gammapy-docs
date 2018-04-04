
# coding: utf-8

# # HGPS
# 
# 
# ## HGPS
# 
# The **H.E.S.S. Galactic Plane Survey (HGPS)** is the first deep and wide survey of the Milky Way in TeV gamma-rays.
# 
# The release data consistes of 10 survey maps and several tables that contain a source catalog and other information.
# 
# The data is available here for download, in FITS format:
# https://www.mpi-hd.mpg.de/hfm/HESS/hgps
# 
# **Please read the Appendix A of the paper to learn about the caveats to using the HGPS data. Especially note that the HGPS survey maps are correlated and thus no detailed source morphology analysis is possible, and also note the caveats concerning spectral models and spectral flux points.**
# 
# ## Topics
# 
# This is a Jupyter notebook that illustrates how to work with the HGPS data from Python. You will learn how to access the HGPS images as well as the HGPS catalog and other tabular data using Astropy and Gammapy.
# 
# * In the first part we will only use Astropy to do some basic things.
# * Then in the second part we'll use Gammapy to do some things that are a little more advanced.
# 
# Note that there are other tools to work with FITS data that we don't explain here. Specifically [DS9](http://ds9.si.edu/) and [Aladin](http://aladin.u-strasbg.fr/) are good FITS image viewers, and [TOPCAT](http://www.star.bris.ac.uk/~mbt/topcat/) is great for FITS tables. Astropy and Gammapy are just one way to work with the HGPS data; any tool that can access FITS data can be used.
# 
# ## Packages
# 
# We will be using the following Python packages and features:
# 
# * [astropy](http://docs.astropy.org/), specifically [astropy.io.fits](http://docs.astropy.org/en/stable/io/fits/index.html) to read the FITS data, [astropy.table.Table](http://docs.astropy.org/en/stable/table/index.html) to work with the tables, but also [astropy.coordinates.SkyCoord](http://docs.astropy.org/en/stable/coordinates/index.html) and [astropy.wcs.WCS](http://docs.astropy.org/en/stable/wcs/index.html) to work with sky and pixel coordinates and [astropy.units.Quantity](http://docs.astropy.org/en/stable/units/index.html) to work with quantities.
# 
# * [regions](https://astropy-regions.readthedocs.io/) to show HGPS source spectral extraction circular regions and create corresponding DS9 regions.
# 
# * [gammapy](http://docs.gammapy.org/), specifically [gammapy.maps](http://docs.gammapy.org/dev/maps/index.html) to work with the HGPS sky maps, and [gammapy.catalog.SourceCatalogHGPS](http://docs.gammapy.org/dev/api/gammapy.catalog.SourceCatalogHGPS.html) and [gammapy.catalog.SourceCatalogObjectHGPS](http://docs.gammapy.org/dev/api/gammapy.catalog.SourceCatalogObjectHGPS.html) to work with the HGPS catalog data, especially the HGPS spectral data using [gammapy.spectrum.models.SpectralModel](http://docs.gammapy.org/dev/api/gammapy.spectrum.models.SpectralModel.html) and [gammapy.spectrum.FluxPoints](http://docs.gammapy.org/dev/api/gammapy.spectrum.FluxPoints.html) objects.
# 
# * [matplotlib](https://matplotlib.org/) for plotting, used via [astropy.visualization](http://docs.astropy.org/en/stable/visualization/index.html) and [gammapy.maps.WcsNDMap.plot](http://docs.gammapy.org/dev/api/gammapy.maps.WcsNDMap.html#gammapy.maps.WcsNDMap.plot)
# for sky map plotting.
# 
# If you're not familiar with Python, Numpy, Astropy, Gammapy or matplotlib yet, you can learn about them using the links to the documentation that we just mentioned, or using the tutorial introductions as explained [here](http://docs.gammapy.org/dev/tutorials.html).
# 

# ## Setup
# 
# We start by importing everything we will use in this notebook, and configuring the notebook to show plots inline.
# If you get an error here, you probably have to install the missing package and re-start the notebook.

# In[1]:


import os

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import regions

from gammapy.maps import Map
from gammapy.catalog import SourceCatalogHGPS


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Data
# 
# First, you need to download the HGPS FITS data from https://www.mpi-hd.mpg.de/hfm/HESS/hgps .
# 
# If you haven't already, you can use the following commands to download the files to your local working directory.
# 
# You don't have to read the code in the next cell; that's just how to downlaod files from Python.
# You could also download the files with your web browser, or from the command line e.g. with curl:
# 
#     mkdir hgps_data
#     cd hgps_data
#     curl -O https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/hgps_catalog_v1.fits.gz
#     curl -O https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/hgps_map_significance_0.1deg_v1.fits.gz
# 
# The rest of this notebook assumes that you have the data files at ``hgps_data_path``.

# In[3]:


# Download HGPS data used in this tutorial to a folder of your choice
# The default `hgps_data` used here is a sub-folder in your current
# working directory (where you started the notebook)
hgps_data_path = 'hgps_data'    

# The Python standard library has a function to retrieve file
# from a given URL; it's just located in a different place
# in Python 3 and Python 2, so while Python 2 is still in use
# we put this to make it work in both versions
try:
    from urllib.request import urlretrieve
except:
    from urllib import urlretrieve

def hgps_data_download():
    base_url = 'https://www.mpi-hd.mpg.de/hfm/HESS/hgps/data/'
    filenames = [
        'hgps_catalog_v1.fits.gz',
        'hgps_map_significance_0.1deg_v1.fits.gz',
    ]
    
    if not os.path.exists(hgps_data_path):
        os.makedirs(hgps_data_path)
    
    for filename in filenames:
        url = base_url + filename
        path = os.path.join(hgps_data_path, filename)
        if os.path.exists(path):
            print('Already downloaded: {}'.format(path))
        else:
            print('Downloading {} to {}'.format(url, path))
            urlretrieve(url, path)

hgps_data_download()

print('\n\nFiles at {} :\n'.format(os.path.abspath(hgps_data_path)))
for filename in os.listdir(hgps_data_path):
    print(filename)


# ## Catalog with Astropy
# 
# tbd

# In[4]:


filename = os.path.join(hgps_data_path, 'hgps_catalog_v1.fits.gz')
hdu_list = fits.open(filename)
hdu_list.info()


# ## Maps with Astropy
# 
# tbd

# In[5]:


filename = os.path.join(hgps_data_path, 'hgps_map_significance_0.1deg_v1.fits.gz')
hdu_list = fits.open(filename)
hdu_list.info()
hdu = hdu_list[0]


# In[6]:


type(hdu.data)


# In[7]:


hdu.data.shape


# In[8]:


hdu.data.max()


# In[9]:


hdu.header


# In[10]:


wcs = WCS(hdu.header)
wcs


# In[11]:


pos = SkyCoord(0, 0, unit='deg', frame='galactic')
pos.to_pixel(wcs)


# ## Convert catalog format
# 
# The HGPS catalog is only released in FITS format.
# 
# This section shows you how you can convert part of the information to the following commonly used formats:
# 
# * CSV
# * DS9 regions
# 
# tbd

# ## Catalog with Gammapy
# 
# tbd

# In[12]:


filename = os.path.join(hgps_data_path, 'hgps_catalog_v1.fits.gz')
cat = SourceCatalogHGPS(filename)


# In[13]:


source = cat[0]


# In[14]:


print(source)


# In[15]:


print(source.spectral_model())


# In[16]:


source.flux_points.table


# In[17]:


source.spectral_model().plot(source.energy_range)
source.spectral_model().plot_error(source.energy_range)
source.flux_points.plot()


# In[18]:


source.components


# ## Maps with Gammapy
# 
# This section shows you how to ... <TODO>

# In[19]:


filename = os.path.join(hgps_data_path, 'hgps_map_significance_0.1deg_v1.fits.gz')
survey_map = Map.read(filename)
survey_map.get_by_coord((0, 0))


# In[20]:


# TODO: show how to make a cutout and plot it
image = survey_map.crop(10)
image.plot()


# In[21]:


# TODO: make model image and compare with data


# ## Conclusions
# 
# This concludes this tutorial how to access and work with the HGPS data from Python, using Astropy and Gammapy.
# 
# There are many things we didn't cover here. For example the [spectrum_models](http://docs.gammapy.org/dev/notebooks/spectrum_models.html) tutorial explains how to define a user-defined spectral model, e.g. you could call one of the astrophysical emission models from [Naima](http://naima.readthedocs.io/). The [sed_fitting_gammacat_fermi](http://docs.gammapy.org/dev/notebooks/sed_fitting_gammacat_fermi.html) tutorial shows how to access other catalog data (e.g. from Fermi-LAT or gamma-cat) and fit models to spectral points.
# 
# 
# * If you have any questions about the HGPS data, please use the contact given at https://www.mpi-hd.mpg.de/hfm/HESS/hgps/ .
# * If you have any questions or issues about Astropy or Gammapy, please use the Gammapy mailing list (see http://gammapy.org/contact.html).
# 
# **Please read the Appendix A of the paper to learn about the caveats to using the HGPS data. Especially note that the HGPS survey maps are correlated and thus no detailed source morphology analysis is possible, and also note the caveats concerning spectral models and spectral flux points.**
