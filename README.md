# Biterm Topic Model

[![CircleCI](https://circleci.com/gh/maximtrp/bitermplus.svg?style=shield)](https://circleci.com/gh/maximtrp/bitermplus)
[![Documentation Status](https://readthedocs.org/projects/bitermplus/badge/?version=latest)](https://bitermplus.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/bitermplus)](https://pepy.tech/project/bitermplus)
![PyPI](https://img.shields.io/pypi/v/bitermplus)

*Bitermplus* implements [Biterm topic model](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) for short texts introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng. Actually, it is a cythonized version of [BTM](https://github.com/xiaohuiyan/BTM). This package is also capable of computing *perplexity* and *semantic coherence* metrics.

## Development

Please note that bitermplus is actively improved.
Refer to [documentation](https://bitermplus.readthedocs.io) to stay up to date.

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
import pandas as pd

# Importing data
df = pd.read_csv(
    'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
texts = df['texts'].str.strip().tolist()

# Vectorizing documents, obtaining full vocabulary and biterms
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
biterms = btm.get_biterms(docs_vec)

# Initializing and running model
model = btm.BTM(
    X, vocabulary, seed=12321, T=8, W=vocabulary.size, M=20, alpha=50/8, beta=0.01)
model.fit(biterms, iterations=20)
p_zd = model.transform(docs_vec)

# Calculating metrics
perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
# or
perplexity = model.perplexity_
coherence = model.coherence_
```
