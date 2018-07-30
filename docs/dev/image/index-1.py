from gammapy.maps import Map
filename = '$GAMMAPY_EXTRA/datasets/fermi_2fhl/fermi_2fhl_vela.fits.gz'
image = Map.read(filename, hdu=2)
image.plot()