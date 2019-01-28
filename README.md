# gammapy-docs

This repository is for the Gammapy documentation build and deploy on Github pages

It does **not** contain the sources for the Gammmapy documentation.
Those are in the `docs` folder of the `gammapy` code repository.

## Overview

The `docs` folder of the `master` branch in this repo
gets served at https://gammapy.github.io/gammapy-docs/

It must contain the rendered HTML Gammapy Sphinx docs
in sub-folders, one for each version.

Example: http://gammapy.github.io/gammapy-docs/0.6

Special versions:

* `docs/dev` - the development version
* `docs/stable/index.html` - forwards to latest stable, e.g. `0.6`
* `docs/index.html` - forwards to `docs/stable`

## Howto


## Update dev version

```
cd build/dev/gammapy
git clean -fdx
git pull
python setup.py develop
time make docs-all
cd ../../..
rm -r docs/dev
cp -r build/dev/gammapy/docs/_build/html docs/dev
git add docs/dev
git commit -m 'update docs/dev'
git push
```

## Update a stable version

```
cd build
mkdir 0.10  # or whatever the version is
cd 0.10
git clone https://github.com/gammapy/gammapy.git
cd gammapy
git checkout v0.10
python setup.py develop
time make docs-all release=v0.10
cd ../../..
cp -r build/0.10/gammapy/docs/_build/html docs/0.10
git add docs/0.10
git commit -m 'Add docs/0.10'
```

Then update `stable/index.html` to point to the new stable version.

## Very old versions

An archive of very old versions of built Gammapy docs is available here:
https://github.com/cdeil/gammapy-docs-rtd-archive

## TODO

* How to fetch the right version of `gammapy-extra`?
* How to set up a conda env for older versions?
* How to avoid the repo from growing forever, i.e. discarding old committed versions in `docs/dev`?

## Notes

* Gammapy v0.6 build doesn't work
