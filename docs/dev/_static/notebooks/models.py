#!/usr/bin/env python
# coding: utf-8

# # Models Gallery
# 
# 
# This is an overview of the Gammapy built-in models in `~gammapy.modeling.models`. Gammapy works with 3D model objects, a `SkyModel(spectral_model, spatial_model)` can represent models with a spectral component and a spatial component while a `SkyDiffuseCube` represent a fully 3D cube template. In the following we are going to see how to create these models and learn more about their specific functionnalities.
# 
# Note that there is a separate tutorial [modeling](modeling.ipynb) that explains about `~gammapy.modeling`,
# the Gammapy modeling and fitting framework. You have to read that to learn how to work with models in order to analyse data.
# 
# Topics covered here:
# 
# - How to create spatial, and spectral models.
# - How to create 3D models and other compound models.
# - How to use the model registries to list all available models or add models.
# - How to work with user defined models for simulations and fitting.
# - How to serialize/read and deserialize/write models.
# 
# ## Setup
# 
# As always, let's get started with some setup ...

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import Angle
from gammapy.maps import Map, WcsGeom
import gammapy.modeling.models as gm
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    SpectralModel,
    PowerLawSpectralModel,
    SkyModels,
    SkyModel,
    SkyDiffuseCube,
)


# ## Models Registries
# 
# The list of built-in models directly availables are given in the models registries: `SPECTRAL_MODELS` and `SPECTRAL_MODELS`. In the custom model section we will see how to add user defined models to these lists.

# In[ ]:


from gammapy.modeling.models import SPECTRAL_MODELS, SPATIAL_MODELS

SPATIAL_MODELS


# In[ ]:


SPECTRAL_MODELS


# To learn more about the definition and parameters of each model have a look to the documentation pages [here](https://docs.gammapy.org/0.15/modeling/index.html#module-gammapy.modeling.models)

# ## Spatial models

# ### Defining and evaluating a spatial model
# 
# Here is an example that shows how to define a Gaussian spatial model:

# In[ ]:


gaussian = gm.GaussianSpatialModel(
    lon_0="2 deg",
    lat_0="2 deg",
    sigma="1 deg",
    e=0.7,
    phi="30 deg",
    frame="galactic",
)


# In order to display the spatial model we can define a map geometry with `WcsGeom`, evaluate the model toward its coordinates and then create a `Map` to plot.

# In[ ]:


# create the geometry
m_geom = WcsGeom.create(
    binsz=0.01, width=(5, 5), skydir=(2, 2), frame="galactic", proj="AIT"
)
coords = m_geom.get_coord()

# evaluate the model
values = gaussian(coords.lon, coords.lat)

# create and plot the map
skymap = Map.from_geom(m_geom, data=values.value, unit=values.unit)
_, ax, _ = skymap.plot()

# then we can do some extra plotting on the image
transform = ax.get_transform("galactic")
phi = gaussian.phi.quantity
ax.scatter(2, 2, transform=transform, s=20, edgecolor="red", facecolor="red")
ax.text(1.5, 1.85, r"$(l_0, b_0)$", transform=transform, ha="center")
ax.plot(
    [2, 2 + np.sin(phi)], [2, 2 + np.cos(phi)], color="r", transform=transform
)
ax.vlines(x=2, color="r", linestyle="--", transform=transform, ymin=-5, ymax=5)
ax.text(2.25, 2.45, r"$\phi$", transform=transform)
ax.contour(skymap.data, cmap="coolwarm", levels=10, alpha=0.6)


# We can do the same with a disk model:

# In[ ]:


disk = gm.DiskSpatialModel(
    lon_0="2 deg",
    lat_0="2 deg",
    r_0="1 deg",
    e=0.8,
    phi="30 deg",
    frame="galactic",
)

m_geom = WcsGeom.create(
    binsz=0.01, width=(3, 3), skydir=(2, 2), frame="galactic", proj="AIT"
)
coords = m_geom.get_coord()
vals = disk(coords.lon, coords.lat)
skymap = Map.from_geom(m_geom, data=vals.value)

_, ax, _ = skymap.plot()

transform = ax.get_transform("galactic")
phi = disk.phi.quantity
ax.scatter(2, 2, transform=transform, s=20, edgecolor="red", facecolor="red")
ax.text(1.7, 1.85, r"$(l_0, b_0)$", transform=transform, ha="center")
ax.plot(
    [2, 2 + np.sin(np.pi / 6)],
    [2, 2 + np.cos(np.pi / 6)],
    color="r",
    transform=transform,
)
ax.vlines(x=2, color="r", linestyle="--", transform=transform, ymin=0, ymax=5)
ax.text(2.15, 2.3, r"$\phi$", transform=transform)


# ### Astropy-regions and ds9 region files
# 
# The spatial model can be exported to [astropy-regions](https://astropy-regions.readthedocs.io/en/latest/) objects that provide several convenient function.
# 

# In[ ]:


print(disk.to_region())


# In particular we can save these regions as [ds9-regions](http://ds9.si.edu/doc/ref/region.html) files.
# Here is an example that shows how to write a ds9 region file for the Fermi-LAT extended source defined in 3FHL catalogue. 

# In[ ]:


from gammapy.catalog import SourceCatalog3FHL
from regions import write_ds9

FERMI_3FHL = SourceCatalog3FHL()
models_reg = [source.spatial_model().to_region() for source in FERMI_3FHL if not source.is_pointlike]
regions = [_ for _ in models_reg if _ is not None]

filename = "./3fhl_extended_shapes.reg"
write_ds9(regions, filename, coordsys="galactic", fmt=".4f", radunit="deg")


# Note that for the parametric models we display the corresonding shape but for the template models we give only the boundary of the map.
# 
# Similarly the position error of the spatial model is described by an astropy-regions object and can be saved to a ds9 regions file.
# 

# In[ ]:


pos_err = FERMI_3FHL["Crab Nebula"].spatial_model().position_error
print(pos_err)


# In[ ]:


regiosn = [pos_err]
filename = "./3fhl_position_error.reg"
write_ds9(regions, filename, coordsys="galactic", fmt=".4f", radunit="deg")


# ## Spectral models
# 
# ### Defining and evaluating a spectral model
# 
# Here are some examples of the built-in spectral models:

# In[ ]:


energy_range = [0.1, 100] * u.TeV

pwl = gm.PowerLawSpectralModel(
    index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
pwl.plot(energy_range, label="pwl")

pwl2 = gm.PowerLaw2SpectralModel(
    amplitude="1e-12 cm-2 s-1", index=2.0, emin="0.1 TeV", emax="100 TeV"
)
pwl2.plot(energy_range, label="pwl2")

ecpl = gm.ExpCutoffPowerLawSpectralModel(
    index=1.5,
    amplitude="1e-12 cm-2 s-1 TeV-1",
    reference="1 TeV",
    lambda_="0.1 TeV-1",
    alpha=1.0,
)
ecpl.plot(energy_range, label="ecpl")

log_parabola = gm.LogParabolaSpectralModel(
    amplitude="1e-12 cm-2 s-1 TeV-1", reference="10 TeV", alpha=2.0, beta=1.0
)
log_parabola.plot(energy_range, label="log_parabola")

plt.ylim(1e-18, 1e-10)
plt.legend();


# You can evaluate a model values with energies given as an astropy `Quantity`.

# In[ ]:


value = pwl2(1 * u.TeV)
values = pwl2([1, 10, 100] * u.TeV)
print(values)
type(values)


# You can also return the energy corresponding to a given flux value of the spectral model with the `inverse` method

# In[ ]:


pwl2.inverse(values)


# In order to integrate the spectral model within an energy range you can use the `integral` method

# In[ ]:


pwl2.integral(0.1 * u.TeV, 100 * u.TeV)


# ### Naima models
# 
# Additionnal gammapy provide an interface to work with [Naima models](https://naima.readthedocs.io/en/latest/api-models.html). 
# In the following we show as an example how to create and plot a spectral model that convolves an `ExpCutoffPowerLawSpectralModel` electron distribution with an `InverseCompton` radiative model, in the presence of multiple seed photon fields.
# 

# In[ ]:


import naima

particle_distribution = naima.models.ExponentialCutoffPowerLaw(
    1e30 / u.eV, 10 * u.TeV, 3.0, 30 * u.TeV
)
radiative_model = naima.radiative.InverseCompton(
    particle_distribution,
    seed_photon_fields=["CMB", ["FIR", 26.5 * u.K, 0.415 * u.eV / u.cm ** 3]],
    Eemin=100 * u.GeV,
)

model = gm.NaimaSpectralModel(radiative_model, distance=1.5 * u.kpc)

opts = {
    "energy_range": [10 * u.GeV, 80 * u.TeV],
    "energy_power": 2,
    "flux_unit": "erg-1 cm-2 s-1",
}

# Plot the total inverse Compton emission
model.plot(label="IC (total)", **opts)

# Plot the separate contributions from each seed photon field
for seed, ls in zip(["CMB", "FIR"], ["-", "--"]):
    model = gm.NaimaSpectralModel(
        radiative_model, seed=seed, distance=1.5 * u.kpc
    )
    model.plot(label=f"IC ({seed})", ls=ls, color="gray", **opts)

plt.legend(loc="best");


# ### EBL absorption models
# 
# Here we illustrate how to create and plot EBL absorption models for a redshift of 0.5:

# In[ ]:


redshift = 0.5
dominguez = gm.Absorption.read_builtin("dominguez").table_model(redshift)
franceschini = gm.Absorption.read_builtin("franceschini").table_model(redshift)
finke = gm.Absorption.read_builtin("finke").table_model(redshift)

plt.figure()
energy_range = [0.08, 3] * u.TeV
opts = dict(energy_range=energy_range, energy_unit="TeV", flux_unit="")
franceschini.plot(label="Franceschini 2008", **opts)
finke.plot(label="Finke 2010", **opts)
dominguez.plot(label="Dominguez 2011", **opts)

plt.ylabel(r"Absorption coefficient [$\exp{(-\tau(E))}$]")
plt.xlim(energy_range.value)
plt.ylim(1e-4, 2)
plt.title(f"EBL models (z={redshift})")
plt.grid(which="both")
plt.legend(loc="best");


# ## Custom models
# 
# In order to add a user defined spectral model you have to create a SpectralModel subclass.
# This new model class should include:
# 
# - a tag used for serialization (it can be the same as the class name)
# - an instantiation of each Parameter with their unit, default values and frozen status
# - the evaluate function where the mathematical expression for the model is defined.
# 
# As an example we will use a PowerLawSpectralModel plus a Gaussian (with fixed width).
# First we define the new custom model class that we name `PLG`:

# In[ ]:


class PLG(SpectralModel):
    tag = "PLG"
    index = Parameter("index", 2, min=0)
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1", min=0)
    reference = Parameter("reference", "1 TeV", frozen=True)
    mean = Parameter("mean", "1 TeV", min=0)
    width = Parameter("width", "0.1 TeV", min=0, frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, mean, width):
        pwl = PowerLawSpectralModel.evaluate(
            energy=energy,
            index=index,
            amplitude=amplitude,
            reference=reference,
        )
        gauss = amplitude * np.exp(-((energy - mean) ** 2) / (2 * width ** 2))
        return pwl + gauss


# then we add it to the spectral model registry so it can be used for fitting and serialization:

# In[ ]:


SPECTRAL_MODELS.append(PLG)


# In[ ]:


custom_model = PLG(
    index=2,
    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
    reference=1 * u.TeV,
    mean=5 * u.TeV,
    width=0.2 * u.TeV,
)
print(custom_model)


# In[ ]:


energy_range = [1, 10] * u.TeV
custom_model.plot(energy_range=energy_range);


# Note that gammapy assumes that all SpectralModel evaluate functions return a flux in unit of "cm-2 s-1 TeV-1" (or equivalent dimensions).
# 
# Similarly you can also create custom spatial models and add them to the `SPATIAL_MODELS` registry. In that case gammapy assumes that the evaluate function return a normalized quantity in "sr-1" such as the model integral over the whole sky is one.
# 
# Once your custom models are defined and added to their model resgistry they can be serialized like the built-in models, as shown at the end of this tutorial.

# ## 3D models and models list
# 
# A source can be modeled by a combination of a spatial and a spectral model using a `SkyModel`. 
# For example we can use the disk and exponential cut-off power-law models defined previously to create a new source model.

# In[ ]:


model1 = SkyModel(spectral_model=ecpl, spatial_model=disk)
print(model1.name)
print(model1)


# Note that for convenience the spatial model component is optionnal. Here we create a source model using only the  power-law model defined previously:
# 

# In[ ]:


model2 = SkyModel(pwl, name="source2")
print(model2.name)
print(model2)


# Additionnaly the `gammapy.modeling.models.SkyDiffuseCube` can be used to represent source models based on templates. It can be created from an existing FITS file:
# 

# In[ ]:


diffuse = SkyDiffuseCube.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz"
)
print(diffuse.name)
print(diffuse)


# A you can see in the previous examples the `name` arugment is also optionnal. However if you want to build complex models you have to define it, so the different sources or backgrounds can be identified (for now the serialisation rely on unique model names).
# 
# Then the global model of an analysis can be define by combining several 3D models (`SkyModel` or `SkyDiffuseCube`)  into a `SkyModels`.
# 
# 

# In[ ]:


models = SkyModels([model1, model2, diffuse])


# Alternatively you can combine the 3D models using the `+` operator.

# In[ ]:


models = model1 + model2 + diffuse


# Note that a `SkyModel` object can be evaluated for a given longitude, latitude, and energy, but the `SkyModels` object cannot. 
# This `SkyModels` container object will be assigned to `Dataset` or `Datasets` together with the data to be fitted as explained in other analysis tutotials (see for example the [modeling](modeling.ipynb) notebook).
# 
# 
# ## Serialization
# 
# The list of models contained in a `SkyModels` object can be exported/imported using yaml configuration files.
# 
# 

# In[ ]:


models_yaml = models.to_yaml()
print(models_yaml)


# The structure of the yaml files follows the structure of the python objects.
# The `components` listed correspond to the `SkyModel` and `SkyDiffuseCube` components of the `SkyModels`. 
# For each `SkyModel` we have  informations about its `name`, `type` (corresponding to the tag attribute) and sub-mobels (i.e `spectral` model and eventually `spatial` model). Then the spatial and spectral models are defiend by their type and parameters. The `parameters` keys name/value/unit are mandatory, while the keys min/max/frozen are optionnals (so you can prepare shorter files).
# 
# If you want to write this list of models to disk and read it back later you can use:

# In[ ]:


models.write("models.yaml", overwrite=True)
models_read = SkyModels.read("models.yaml")


# Additionnaly the models can exported and imported togeter with the data using the `Datasets.read()` and `Datasets.write()` methods as shown in the [analysis_mwl](analysis_mwl) notebook.

# In[ ]:




