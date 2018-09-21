
# coding: utf-8

# # H.E.S.S. with Gammapy
# 
# This tutorial explains how to analyse [H.E.S.S.](https://www.mpi-hd.mpg.de/hfm/HESS) data with Gammapy.
# 
# We will analyse four observation runs of the Crab nebula, which are part of the [H.E.S.S. first public test data release](https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/). We will use the template background model from off runs produced in the [background_model.ipynb](background_model.ipynb) notebook with Gammapy.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


# TODO: Write this notebook when this issue is fixed:
# https://github.com/gammapy/gammapy/issues/1819
# from gammapy.data import DataStore
# data_store = DataStore.from_dir(
#     base_dir="",
#     hdu_table_filename="$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz",
#     obs_table_filename="$GAMMAPY_DATA/hess-dl3-dr1/hess-dl3-dr3-with-background.fits.gz",
# )
# data_store.info()

