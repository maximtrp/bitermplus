# Biterm Topic Model

[![CircleCI](https://circleci.com/gh/maximtrp/bitermplus.svg?style=shield)](https://circleci.com/gh/maximtrp/bitermplus)

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

## Example

```python
import bitermplus as btm
import numpy as np
from gzip import open as gzip_open
from sklearn.feature_extraction.text import CountVectorizer

# Importing and vectorizing text data
with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
    texts = file.readlines()

vec = CountVectorizer(lowercase=False)
X = vec.fit_transform(texts)

# Getting full vocabulary and creating biterms
vocab = np.array(vec.get_feature_names())
biterms = btm.util.biterms(X)

# Initializing and running model
model = btm.BTM(8, vocab.size, alpha=50/8, beta=0.01, L=0.5)
model.fit(biterms, iterations=10)
P_zd = model.transform(biterms)

# Calculating metrics
perplexity = btm.metrics.perplexity(model.phi_, P_zd, X, 8)
coherence = btm.metrics.coherence(model.phi_, X, M=20)
```
