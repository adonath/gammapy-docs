#!/usr/bin/env python
# coding: utf-8

# # Source catalogs
# 
# **TODO: write me!**
# 
# This is a hands-on tutorial introduction to `~gammapy.catalog`.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from gammapy.catalog import SOURCE_CATALOGS


# In[ ]:


SOURCE_CATALOGS


# In[ ]:


catalog = SOURCE_CATALOGS["3fgl"]()
source = catalog["3FGL J0349.9-2102"]


# In[ ]:


lc = source.lightcurve
lc.plot()


# In[ ]:





# In[ ]:




