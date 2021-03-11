# Biterm Topic Model

[![CircleCI](https://circleci.com/gh/maximtrp/bitermplus.svg?style=shield)](https://circleci.com/gh/maximtrp/bitermplus)
[![Documentation Status](https://readthedocs.org/projects/bitermplus/badge/?version=latest)](https://bitermplus.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/bitermplus)](https://pepy.tech/project/bitermplus)
![PyPI](https://img.shields.io/pypi/v/bitermplus)

*Bitermplus* implements [Biterm topic model](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) for short texts introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng. It uses Cython for better performance and includes two implementations of BTM: 1) `bitermplus.BTM` class (based on [BTM](https://github.com/xiaohuiyan/BTM) by Xiaohui Yan), 2) `bitermplus.btm_depr.BTM` class (based on [biterm](https://github.com/markoarnauto/biterm) package by Markus Tretzmüller). This package is also capable of computing *perplexity* and *semantic coherence* metrics.

## Requirements

* Cython
* NumPy
* Pandas
* SciPy
* Scikit-learn
* pyLDAvis (optional)

## Setup

### Linux and Windows

There should be no issues with installing *bitermplus* under these OSes. You can install the package directly from PyPi.

```bash
pip install bitermplus
```

Or from this repo:

```bash
pip install git+https://github.com/maximtrp/bitermplus.git
```

### Mac OS

First, you need to install XCode CLT and [Homebrew](https://brew.sh).
Then, install `libomp` using `brew`:

```bash
xcode-select --install
brew install libomp
pip3 install bitermplus
```

## Example

```python
import bitermplus as btm
import numpy as np
from gzip import open as gzip_open

# Importing and vectorizing text data
with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
    texts = file.readlines()

# Vectorizing documents, obtaining full vocabulary and biterms
X, vocab = btm.util.get_words_freqs(texts)
docs_vec = btm.util.get_vectorized_docs(X)
biterms = btm.util.get_biterms(X)

# Initializing and running model
model = btm.BTM(X, T=8, W=vocab.size, M=20, alpha=50/8, beta=0.01)
model.fit(biterms, iterations=20)
p_zd = model.transform(docs_vec)

# Calculating metrics
perplexity = btm.metrics.perplexity(model.matrix_words_topics_, p_zd, X, 8)
coherence = btm.metrics.coherence(model.matrix_words_topics_, X, M=20)
# or
perplexity = model.perplexity_
coherence = model.coherence_
```

## Acknowledgement

Markus Tretzmüller [@markoarnauto](https://github.com/markoarnauto)
