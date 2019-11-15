#!/usr/bin/env python
# coding: utf-8

# # Source catalogs
# 
# **TODO: write me!**
# 
# This is a hands-on tutorial introduction to `~gammapy.catalog`.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


from gammapy.catalog import SOURCE_CATALOGS


# In[ ]:


SOURCE_CATALOGS


# In[ ]:


catalog = SOURCE_CATALOGS["3fgl"]()
source = catalog["3FGL J0349.9-2102"]


# In[ ]:


# Example to show how to plot spectrum of Fermi/LAT sources.


def plot_source_spectrum(source, label, color):
    opts = dict(energy_power=2, flux_unit="erg-1 cm-2 s-1")
    spec = source.spectral_model()
    spec.plot(
        energy_range=source.energy_range, label=label, color=color, **opts
    )
    spec.plot_error(energy_range=source.energy_range, facecolor=color, **opts)
    source.flux_points.to_sed_type("dnde").plot(color=color, **opts)


def plot_source_spectra(name):
    plot_source_spectrum(
        source=SOURCE_CATALOGS["3fgl"]()[name], label="3FGL", color="r"
    )
    plot_source_spectrum(
        source=SOURCE_CATALOGS["2fhl"]()[name], label="2FHL", color="g"
    )
    plot_source_spectrum(
        source=SOURCE_CATALOGS["3fhl"]()[name], label="3FHL", color="b"
    )

    ax = plt.gca()
    ax.set_ylim(1.0e-12, 7.0e-11)
    ax.set_xlim(1.0e-4, 2.0)
    ax.set_xlabel("Energy (TeV)")
    ax.set_ylabel("E^2 dN/dE (erg cm-2 s-1])")
    plt.legend(loc=0)


# Select your favourite source
# (must be named like this in ASSOC columns of all catalogs)
name = "PKS 2155-304"
plot_source_spectra(name)


# In[ ]:


lc = source.lightcurve
lc.plot()


# In[ ]:




