# Biterm Topic Model

![CircleCI](https://img.shields.io/circleci/build/github/maximtrp/bitermplus)

This package implements [Biterm topic model](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) for short texts introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng. It is based on [biterm](https://github.com/markoarnauto/biterm) package by [@markoarnauto](https://github.com/markoarnauto). Unfortunately, *biterm* package is not maintained anymore.

*Bitermplus* is a fixed and optimized successor. Pure Python version of `BTM` class was removed. Class `oBTM` was strongly optimized using typed memoryviews in Cython and now replaces `BTM` class.

## Setup

You can install the package from PyPi:

```bash
pip install bitermplus
```

Or from this repo:

```bash
pip install git+https://github.com/maximtrp/bitermplus.git
```

## Examples

