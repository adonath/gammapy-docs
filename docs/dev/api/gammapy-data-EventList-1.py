import matplotlib.pyplot as plt
from gammapy.data import EventList

events = EventList.read('$GAMMAPY_EXTRA/datasets/hess-dl3-dr1//data/hess_dl3_dr1_obs_id_023523.fits.gz')
events.plot_time()
plt.show()