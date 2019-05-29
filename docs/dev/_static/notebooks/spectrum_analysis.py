
# coding: utf-8

# # Spectral analysis with Gammapy

# ## Introduction
# 
# This notebook explains in detail how to use the classes in [gammapy.spectrum](https://docs.gammapy.org/dev/spectrum/index.html) and related ones.
# 
# Based on a datasets of 4 Crab observations with H.E.S.S. (simulated events for now) we will perform a full region based spectral analysis, i.e. extracting source and background counts from certain 
# regions, and fitting them using the forward-folding approach. We will use the following classes
# 
# Data handling:
# 
# * [gammapy.data.DataStore](https://docs.gammapy.org/dev/api/gammapy.data.DataStore.html)
# * [gammapy.data.DataStoreObservation](https://docs.gammapy.org/dev/api/gammapy.data.DataStoreObservation.html)
# * [gammapy.data.ObservationStats](https://docs.gammapy.org/dev/api/gammapy.data.ObservationStats.html)
# * [gammapy.data.ObservationSummary](https://docs.gammapy.org/dev/api/gammapy.data.ObservationSummary.html)
# 
# To extract the 1-dim spectral information:
# 
# * [gammapy.spectrum.SpectrumExtraction](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumExtraction.html)
# * [gammapy.background.ReflectedRegionsBackgroundEstimator](https://docs.gammapy.org/dev/api/gammapy.background.ReflectedRegionsBackgroundEstimator.html)
# 
# To perform the joint fit: 
# * [gammapy.spectrum.SpectrumDatasetOnOff](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumDatasetOnOff.html)
# * [gammapy.spectrum.SpectrumDatasetOnOffStacker](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumDatasetOnOffStacker.html)
# 
# * [gammapy.spectrum.models.PowerLaw](https://docs.gammapy.org/dev/api/gammapy.spectrum.models.PowerLaw.html)
# * [gammapy.spectrum.models.ExponentialCutoffPowerLaw](https://docs.gammapy.org/dev/api/gammapy.spectrum.models.ExponentialCutoffPowerLaw.html)
# * [gammapy.spectrum.models.LogParabola](https://docs.gammapy.org/dev/api/gammapy.spectrum.models.LogParabola.html)
# 
# To compute flux points (a.k.a. "SED" = "spectral energy distribution")
# 
# * [gammapy.spectrum.FluxPoints](https://docs.gammapy.org/dev/api/gammapy.spectrum.FluxPoints.html)
# * [gammapy.spectrum.FluxPointsEstimator](https://docs.gammapy.org/dev/api/gammapy.spectrum.FluxPointsEstimator.html)
# 
# Feedback welcome!

# ## Setup
# 
# As usual, we'll start with some setup ...

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# Check package versions
import gammapy
import numpy as np
import astropy
import regions
import sherpa

print("gammapy:", gammapy.__version__)
print("numpy:", np.__version__)
print("astropy", astropy.__version__)
print("regions", regions.__version__)
print("sherpa", sherpa.__version__)


# In[ ]:


import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import vstack as vstack_table
from regions import CircleSkyRegion
from gammapy.data import DataStore
from gammapy.data import ObservationStats, ObservationSummary
from gammapy.background.reflected import ReflectedRegionsBackgroundEstimator
from gammapy.utils.energy import EnergyBounds
from gammapy.spectrum import SpectrumExtraction
from gammapy.spectrum import SpectrumDatasetOnOff
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum import CrabSpectrum
from gammapy.spectrum import FluxPointsEstimator, FluxPointsDataset
from gammapy.maps import Map
from gammapy.utils.fitting import Fit


# ## Configure logger
# 
# Most high level classes in gammapy have the possibility to turn on logging or debug output. We well configure the logger in the following. For more info see https://docs.python.org/2/howto/logging.html#logging-basic-tutorial

# In[ ]:


# Setup the logger
import logging

logging.basicConfig()
logging.getLogger("gammapy.spectrum").setLevel("WARNING")


# ## Load Data
# 
# First, we select and load some H.E.S.S. observations of the Crab nebula (simulated events for now).
# 
# We will access the events, effective area, energy dispersion, livetime and PSF for containement correction.

# In[ ]:


datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
obs_ids = [23523, 23526, 23559, 23592]
observations = datastore.get_observations(obs_ids)


# ## Define Target Region
# 
# The next step is to define a signal extraction region, also known as on region. In the simplest case this is just a [CircleSkyRegion](http://astropy-regions.readthedocs.io/en/latest/api/regions.CircleSkyRegion.html#regions.CircleSkyRegion), but here we will use the ``Target`` class in gammapy that is useful for book-keeping if you run several analysis in a script.

# In[ ]:


target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


# ## Create exclusion mask
# 
# We will use the reflected regions method to place off regions to estimate the background level in the on region.
# To make sure the off regions don't contain gamma-ray emission, we create an exclusion mask.
# 
# Using http://gamma-sky.net/ we find that there's only one known gamma-ray source near the Crab nebula: the AGN called [RGB J0521+212](http://gamma-sky.net/#/cat/tev/23) at GLON = 183.604 deg and GLAT = -8.708 deg.

# In[ ]:


exclusion_region = CircleSkyRegion(
    center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
    radius=0.5 * u.deg,
)

skydir = target_position.galactic
exclusion_mask = Map.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", coordsys="GAL"
)

mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
exclusion_mask.data = mask
exclusion_mask.plot();


# ## Estimate background
# 
# Next we will manually perform a background estimate by placing [reflected regions](https://docs.gammapy.org/dev/background/reflected.html) around the pointing position and looking at the source statistics. This will result in a  [gammapy.background.BackgroundEstimate](https://docs.gammapy.org/dev/api/gammapy.background.BackgroundEstimate.html) that serves as input for other classes in gammapy.

# In[ ]:


background_estimator = ReflectedRegionsBackgroundEstimator(
    observations=observations,
    on_region=on_region,
    exclusion_mask=exclusion_mask,
)

background_estimator.run()


# In[ ]:


#print(background_estimator.result[0])


# In[ ]:


plt.figure(figsize=(8, 8))
background_estimator.plot(add_legend=True);


# ## Source statistic
# 
# Next we're going to look at the overall source statistics in our signal region. For more info about what debug plots you can create check out the [ObservationSummary](https://docs.gammapy.org/dev/api/gammapy.data.ObservationSummary.html#gammapy.data.ObservationSummary) class.

# In[ ]:


stats = []
for obs, bkg in zip(observations, background_estimator.result):
    stats.append(ObservationStats.from_observation(obs, bkg))

print(stats[1])

obs_summary = ObservationSummary(stats)
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(121)

obs_summary.plot_excess_vs_livetime(ax=ax1)
ax2 = fig.add_subplot(122)
obs_summary.plot_significance_vs_livetime(ax=ax2);


# ## Extract spectrum
# 
# Now, we're going to extract a spectrum using the [SpectrumExtraction](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumExtraction.html) class. We provide the reconstructed energy binning we want to use. It is expected to be a Quantity with unit energy, i.e. an array with an energy unit. We use a utility function to create it. We also provide the true energy binning to use.

# In[ ]:


e_reco = EnergyBounds.equal_log_spacing(0.1, 40, 40, unit="TeV")
e_true = EnergyBounds.equal_log_spacing(0.05, 100.0, 200, unit="TeV")


# Instantiate a [SpectrumExtraction](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumExtraction.html) object that will do the extraction. The containment_correction parameter is there to allow for PSF leakage correction if one is working with full enclosure IRFs. We also compute a threshold energy and store the result in OGIP compliant files (pha, rmf, arf). This last step might be omitted though.

# In[ ]:


ANALYSIS_DIR = "crab_analysis"

extraction = SpectrumExtraction(
    observations=observations,
    bkg_estimate=background_estimator.result,
    containment_correction=False,
)
extraction.run()

# Add a condition on correct energy range in case it is not set by default
extraction.compute_energy_threshold(method_lo="area_max", area_percent_lo=10.0)

print(extraction.spectrum_observations[0])
# Write output in the form of OGIP files: PHA, ARF, RMF, BKG
# extraction.write(outdir=ANALYSIS_DIR, overwrite=True)


# ## Look at observations
# 
# Now we will look at the files we just created. We will use the [SpectrumDatasetOnOff](https://docs.gammapy.org/dev/api/gammapy.spectrum.SpectrumDatasetOnOff.html) object that are still in memory from the extraction step. Note, however, that you could also read them from disk if you have written them in the step above. The ``ANALYSIS_DIR`` folder contains 4 ``FITS`` files for each observation. These files are described in detail [here](https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/ogip/index.html). In short, they correspond to the on vector, the off vector, the effectie area, and the energy dispersion.

# In[ ]:


# filename = ANALYSIS_DIR + '/ogip_data/pha_obs23523.fits'
# obs = SpectrumDatasetOnOff.from_ogip_files(filename)

# Requires IPython widgets
# _ = extraction.spectrum_observations.peek()

extraction.spectrum_observations[0].peek()


# ## Fit spectrum
# 
# Now we'll fit a global model to the spectrum. First we do a joint likelihood fit to all observations. If you want to stack the observations see below. We will also produce a debug plot in order to show how the global fit matches one of the individual observations.

# In[ ]:


model = PowerLaw(
    index=2, amplitude=2e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
)

datasets_joint = extraction.spectrum_observations

for dataset in datasets_joint:
    dataset.model = model

fit_joint = Fit(datasets_joint)
result_joint = fit_joint.run()

# we make a copy here to compare it later
model_best_joint = model.copy()


# In[ ]:


print(result_joint)


# In[ ]:


plt.figure(figsize=(8, 6))
ax_spectrum, ax_residual = datasets_joint[0].plot_fit()
ax_spectrum.set_ylim(0, 20)


# ## Compute Flux Points
# 
# To round up our analysis we can compute flux points by fitting the norm of the global model in energy bands. We'll use a fixed energy binning for now.

# In[ ]:


# Define energy binning
from gammapy.spectrum import SpectrumDatasetOnOffStacker

stacker = SpectrumDatasetOnOffStacker(extraction.spectrum_observations)
dataset_stacked = stacker.run()

e_min, e_max = dataset_stacked.counts.lo_threshold.to_value("TeV"), 30
e_edges = np.logspace(np.log10(e_min), np.log10(e_max), 15) * u.TeV


# In[ ]:


dataset_stacked.model = model

fpe = FluxPointsEstimator(datasets=[dataset_stacked], e_edges=e_edges)
flux_points = fpe.run()


# In[ ]:


flux_points.table_formatted


# Now we plot the flux points and their likelihood profiles. For the plotting of upper limits we choose a threshold of TS < 4.

# In[ ]:


plt.figure(figsize=(8, 5))
flux_points.table["is_ul"] = flux_points.table["ts"] < 4
ax = flux_points.plot(
    energy_power=2, flux_unit="erg-1 cm-2 s-1", color="darkorange"
)
flux_points.to_sed_type("e2dnde").plot_likelihood(ax=ax)


# The final plot with the best fit model, flux points and residuals can be quickly made like this: 

# In[ ]:


model.parameters.covariance = result_joint.parameters.covariance
flux_points_dataset = FluxPointsDataset(data=flux_points, model=model)


# In[ ]:


plt.figure(figsize=(8, 6))
flux_points_dataset.peek();


# ## Stack observations
# 
# And alternative approach to fitting the spectrum is stacking all observations first and the fitting a model to the stacked observation. This works as follows. A comparison to the joint likelihood fit is also printed.

# In[ ]:


dataset_stacked.model = model
stacked_fit = Fit([dataset_stacked])
result_stacked = stacked_fit.run()

# make a copy to compare later
model_best_stacked = model.copy()


# In[ ]:


print(result_stacked)


# In[ ]:


model_best_joint.parameters.to_table()


# In[ ]:


model_best_stacked.parameters.to_table()


# Finally, we compare the results of our stacked analysis to a previously published Crab Nebula Spectrum for reference. This is available in `gammapy.spectrum`.

# In[ ]:



model_best_stacked.plot(energy_range=[0.1,30]*u.TeV, label="Stacked analysis result")
model_best_stacked.plot_error(energy_range=[0.1,30]*u.TeV)
CrabSpectrum().model.plot(energy_range=[0.1,30]*u.TeV, label="Crab reference")
plt.legend()


# ## Exercises
# 
# Some things we might do:
# 
# - Fit a different spectral model (ECPL or CPL or ...)
# - Use different method or parameters to compute the flux points
# - Do a chi^2 fit to the flux points and compare
# 
# TODO: give pointers how to do this (and maybe write a notebook with solutions)
