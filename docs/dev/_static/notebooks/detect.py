#!/usr/bin/env python
# coding: utf-8

# # Source detection with Gammapy
# 
# ## Context
# 
# The first task in a source catalogue production is to identify significant excesses in the data that can be associated to unknown sources and provide a preliminary parametrization in term of position, extent, and flux. In this notebook we will use Fermi-LAT data to illustrate how to detect candidate sources in counts images with known background.
# 
# **Objective: build a list of significant excesses in a Fermi-LAT map**
# 
# 
# ## Proposed approach 
# 
# This notebook show how to do source detection with Gammapy using the methods available in `~gammapy.detect`.
# We will use images from a Fermi-LAT 3FHL high-energy Galactic center dataset to do this:
# 
# * perform adaptive smoothing on counts image
# * produce 2-dimensional test-statistics (TS)
# * run a peak finder to detect point-source candidates
# * compute Li & Ma significance images
# * estimate source candidates radius and excess counts
# 
# Note that what we do here is a quick-look analysis, the production of real source catalogs use more elaborate procedures.
# 
# We will work with the following functions and classes:
# 
# * `~gammapy.maps.WcsNDMap`
# * `~gammapy.detect.ASmooth`
# * `~gammapy.detect.TSMapEstimator`
# * `~gammapy.detect.find_peaks`
# * `~gammapy.detect.compute_lima_image`
# 

# ## Setup
# 
# As always, let's get started with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from gammapy.maps import Map
from gammapy.detect import (
    ASmooth,
    TSMapEstimator,
    find_peaks,
    compute_lima_image,
)
from gammapy.catalog import SOURCE_CATALOGS
from gammapy.cube import PSFKernel
from gammapy.stats import significance
from astropy.coordinates import SkyCoord
from astropy.convolution import Tophat2DKernel
import astropy.units as u
import numpy as np


# In[ ]:


# defalut matplotlib colors without grey
colors = [
    u"#1f77b4",
    u"#ff7f0e",
    u"#2ca02c",
    u"#d62728",
    u"#9467bd",
    u"#8c564b",
    u"#e377c2",
    u"#bcbd22",
    u"#17becf",
]


# ## Read in input images
# 
# We first read in the counts cube and sum over the energy axis:

# In[ ]:


counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz")
background = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background.fits.gz"
)
exposure = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure.fits.gz"
)

maps = {"counts": counts, "background": background, "exposure": exposure}

kernel = PSFKernel.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf.fits.gz"
)


# ## Adaptive smoothing
# 
# For visualisation purpose it can be nice to look at a smoothed counts image. This can be performed using the adaptive smoothing algorithm from [Ebeling et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.368...65E/abstract).
# In the following example the `threshold` argument gives the minimum significance expected, values below are clipped.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'scales = u.Quantity(np.arange(0.05, 1, 0.05), unit="deg")\nsmooth = ASmooth(threshold=3, scales=scales)\nimages = smooth.run(**maps)')


# In[ ]:


plt.figure(figsize=(15, 5))
images["counts"].plot(add_cbar=True, vmax=10);


# ## TS map estimation
# 
# The Test Statistic, TS = 2 ∆ log L ([Mattox et al. 1996](https://ui.adsabs.harvard.edu/abs/1996ApJ...461..396M/abstract)), compares the likelihood function L optimized with and without a given source.
# The TS map is computed by fitting by a single amplitude parameter on each pixel as described in Appendix A of [Stewart (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...495..989S/abstract). The fit is simplified by finding roots of the derivative of the fit statistics (default settings use [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method)).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'estimator = TSMapEstimator()\nimages = estimator.run(maps, kernel.data)')


# ### Plot resulting images

# In[ ]:


plt.figure(figsize=(15, 5))
images["sqrt_ts"].plot(add_cbar=True);


# In[ ]:


plt.figure(figsize=(15, 5))
images["flux"].plot(add_cbar=True, stretch="sqrt", vmin=0);


# In[ ]:


plt.figure(figsize=(15, 5))
images["niter"].plot(add_cbar=True);


# ## Source candidates
# 
# Let's run a peak finder on the `sqrt_ts` image to get a list of point-sources candidates (positions and peak `sqrt_ts` values).
# The `find_peaks` function performs a local maximun search in a sliding window, the argument `min_distance` is the minimum pixel distance between peaks (smallest possible value and default is 1 pixel).

# In[ ]:


sources = find_peaks(images["sqrt_ts"], threshold=8, min_distance=1)
nsou = len(sources)
sources


# In[ ]:


# Plot sources on top of significance sky image
plt.figure(figsize=(15, 5))

_, ax, _ = images["sqrt_ts"].plot(add_cbar=True)

ax.scatter(
    sources["ra"],
    sources["dec"],
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="w",
    marker="o",
    s=600,
    lw=1.5,
);


# Note that we used the instrument point-spread-function (PSF) as kernel, so the hypothesis we test is the presence of a point source. In order to test for extended sources we would have to use as kernel an extended template convolved by the PSF. Alternatively, we can compute the significance of an extended excess using the Li & Ma formalism, which is faster as no fitting is involve.

# ## Li & Ma significance maps
# 
# We can compute significance for an observed number of counts and known background using an extension of equation (17) from the [Li & Ma (1983)](https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L/abstract) (see `gammapy.stats.significance` for details). We can perform this calculation intergating the counts within different radius. To do so we use an astropy Tophat kernel with the `compute_lima_image` function.
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'radius = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])\npixsize = counts.geom.pixel_scales[0].value\nnr = len(radius)\nsigni = np.zeros((nsou, nr))\nexcess = np.zeros((nsou, nr))\nfor kr in range(nr):\n    npixel = radius[kr] / pixsize\n    kernel = Tophat2DKernel(npixel)\n    result = compute_lima_image(counts, background, kernel)\n    signi[:, kr] = result["significance"].data[sources["y"], sources["x"]]\n    excess[:, kr] = result["excess"].data[sources["y"], sources["x"]]')


# For simplicity we saved the significance and excess at the position of the candidates found previously on the TS map, but we could aslo have applied the peak finder on these significances maps for each scale, or alternatively implemented a 3D peak detection (in longitude, latitude, radius). Now let's look at the significance versus integration radius:

# In[ ]:


plt.figure()
for ks in range(nsou):
    plt.plot(radius, signi[ks, :], color=colors[ks])
plt.xlabel("Radius")
plt.ylabel("Li & Ma Significance")
plt.title("Guessing optimal radius of each candidate");


# We can add the optimal radius guessed and the corresdponding excess to the source candidate properties table.

# In[ ]:


# rename the value key to sqrt(TS)_PS
sources.rename_column("value", "sqrt(TS)_PS")

index = np.argmax(signi, axis=1)
sources["significance"] = signi[range(nsou), index]
sources["radius"] = radius[index]
sources["excess"] = excess[range(nsou), index]
sources


# In[ ]:


# Plot candidates sources on top of significance sky image with radius guess
plt.figure(figsize=(15, 5))

_, ax, _ = images["sqrt_ts"].plot(add_cbar=True, cmap=cm.Greys_r)

phi = np.arange(0, 2 * np.pi, 0.01)
for ks in range(nsou):
    x = sources["x"][ks] + sources["radius"][ks] / pixsize * np.cos(phi)
    y = sources["y"][ks] + sources["radius"][ks] / pixsize * np.sin(phi)
    ax.plot(x, y, "-", color=colors[ks], lw=1.5);


# Note that the optimal radius of nested sources is likely overestimated due to their neighbor. We limited this example to only the most significant source above ~8 sigma. When lowering the detection threshold the number of candidated increase together with the source confusion.

# ## What next?
# 
# In this notebook, we have seen how to work with images and compute TS and significance images from counts data, if a background estimate is already available.
# 
# Here's some suggestions what to do next:
# 
# - Look how background estimation is performed for IACTs with and without the high-level interface in [analysis_1](analysis_1.ipynb) and [analysis_2](analysis_2.ipynb) notebooks, respectively
# - Learn about 2D model fitting in the [image_analysis](image_analysis.ipynb) notebook
# - find more about Fermi-LAT data analysis in the [fermi_lat](fermi_lat.ipynb) notebook
# - Use source candidates to build a model and perform a 3D fitting (see [analysis_3d](analysis_3d.ipynb), [analysis_mwl](analysis_mwl) notebooks for some hints)

# In[ ]:




