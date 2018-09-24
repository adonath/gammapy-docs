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
* `docs/index.html` - forwards to `docs/dev`

## Howto


## Update dev version

```
cd build/dev/gammapy
git clean -fdx
git pull
python setup.py build_ext -i
time make docs-all
cd ../../..
rm -r docs/dev
cp -r build/dev/gammapy/docs/_build/html docs/dev
git add docs/dev
git commit -m 'update docs/dev'
git push
```

## Update a stable version

Same as before, except check out the right tag and pass it to the `make docs-all` command as a param `release`

```
git checkout v0.6
```

```
time make docs-all release=v0.8
```

## Very old versions

An archive of very old versions of built Gammapy docs is available here:
https://github.com/cdeil/gammapy-docs-rtd-archive

## TODO

* How to fetch the right version of `gammapy-extra`?
* How to set up a conda env for older versions?
* How to avoid the repo from growing forever, i.e. discarding old committed versions in `docs/dev`?

## Notes

* Gammapy v0.6 build doesn't work
