# Scop
[![DOI](https://zenodo.org/badge/DOI/10/gpqm.svg)](https://doi.org/gpqm)

Scop is an opinionated, self-contained data format and post-optimization toolchain. Currently, it's aimed towards use with OpenMDAO.

If you can identify with one of multiple of these statements, you might be interested in Scop:
* You do multi-objective heuristic optimization and just want to see the feasible and non-dominated subset of all your designs.
* You want to do efficient manipulations and calculations across multiple designs and not only look at single designs in isolation.
* You don't necessarily want to persist all data to disk all the time.
* You think that xarray `Dataset`s or Pandas `DataFrame`s have more swag than SQLite blobs.

## How do I get it?
You use Conda, and run the following in your environment:

    conda install ovidner::openmdao-scop

## How do I use it?
Until I write some decent documentation, let me know what you want to do, and I'll try to see if Scop can help you.

## What's up with the name?
It's just a word, because...
* I'm tired of cheesy acronyms.
* Just using a word gives me a greater freedom to pivot the project without necessarily changing its name.

If *Scop* were to be an acronym, it could stand for *Self-Contained Optimization Pile-of-data*.
