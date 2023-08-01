# Scop

[![DOI](https://zenodo.org/badge/DOI/10/gpqm.svg)](https://doi.org/gpqm)

Scop is an opinionated, self-contained data format and toolchain for use in multi-disciplinary design optimization. More specifically, it provides utilities for describing your models and variables prior to optimization, so that you get a neat, `xarray`-based dataset to manipulate after optimization (or DoE) runtime. It also provides utilities for performing common post-optimization tasks.

Scop's pre-optimization utilities are probably most useful in cases where you have many models and variables to manage and especially when your variables might be multi-dimensional.

Scop has been developed under academic-industrial cooperation. Please see the following papers for application examples:

* [Vidner, Olle, Robert Pettersson, Johan A. Persson, and Johan Ölvander. “Multidisciplinary Design Optimization of a Mobile Miner Using the OpenMDAO Platform.” In Proceedings of the Design Society, 1:2207–16. Cambridge University Press, 2021](https://doi.org/10/grrp)
* [Vidner, Olle, Camilla Wehlin, Johan A. Persson, and Johan Ölvander. “Configuring Customized Products with Design Optimization and Value-Driven Design.” In Proceedings of the Design Society, 1:741–50. Cambridge University Press, 2021](https://doi.org/10/grrm)
* [Vidner, Olle, Anton Wiberg, Robert Pettersson, Johan A. Persson, and Johan Ölvander. “Optimization-Based Configuration of Engineer-to-Order Products,” 2023](https://doi.org/10.5281/zenodo.7915357)

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
