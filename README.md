# Biterm Topic Model

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/maximtrp/bitermplus/package-test.yml)
[![Documentation Status](https://readthedocs.org/projects/bitermplus/badge/?version=latest)](https://bitermplus.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/192b6a75449040ff868932a15ca28ce9)](https://www.codacy.com/gh/maximtrp/bitermplus/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=maximtrp/bitermplus&amp;utm_campaign=Badge_Grade)
[![Issues](https://img.shields.io/github/issues/maximtrp/bitermplus.svg)](https://github.com/maximtrp/bitermplus/issues)
[![Downloads](https://static.pepy.tech/badge/bitermplus)](https://pepy.tech/project/bitermplus)
![PyPI](https://img.shields.io/pypi/v/bitermplus)

*Bitermplus* implements [Biterm topic model](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) for short texts introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi Cheng. Actually, it is a cythonized version of [BTM](https://github.com/xiaohuiyan/BTM). This package is also capable of computing *perplexity*, *semantic coherence*, and *entropy* metrics.

## Development

Please note that bitermplus is actively improved.
Refer to [documentation](https://bitermplus.readthedocs.io) to stay up to date.

## Requirements

* cython
* numpy
* pandas
* scipy
* scikit-learn
* tqdm

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

If you have the following issue with libomp (`fatal error: 'omp.h' file not found`), run `brew info libomp` in the console:

```bash
brew info libomp
```

You should see the following output:

```
libomp: stable 15.0.5 (bottled) [keg-only]
LLVM's OpenMP runtime library
https://openmp.llvm.org/
/opt/homebrew/Cellar/libomp/15.0.5 (7 files, 1.6MB)
Poured from bottle on 2022-11-19 at 12:16:49
From: https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/libomp.rb
License: MIT
==> Dependencies
Build: cmake ✘, lit ✘
==> Caveats
libomp is keg-only, which means it was not symlinked into /opt/homebrew,
because it can override GCC headers and result in broken builds.

For compilers to find libomp you may need to set:
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

==> Analytics
install: 192,197 (30 days), 373,389 (90 days), 1,285,192 (365 days)
install-on-request: 24,388 (30 days), 48,013 (90 days), 164,666 (365 days)
build-error: 0 (30 days)
```

Export `LDFLAGS` and `CPPFLAGS` as suggested in brew output:

```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

## Example

### Model fitting

```python
import bitermplus as btm
import numpy as np
import pandas as pd

# IMPORTING DATA
df = pd.read_csv(
    'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
texts = df['texts'].str.strip().tolist()

# PREPROCESSING
# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
tf = np.array(X.sum(axis=0)).ravel()
# Vectorizing documents
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))
# Generating biterms
biterms = btm.get_biterms(docs_vec)

# INITIALIZING AND RUNNING MODEL
model = btm.BTM(
    X, vocabulary, seed=12321, T=8, M=20, alpha=50/8, beta=0.01)
model.fit(biterms, iterations=20)
p_zd = model.transform(docs_vec)

# METRICS
perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
# or
perplexity = model.perplexity_
coherence = model.coherence_

# LABELS
model.labels_
# or
btm.get_docs_top_topic(texts, model.matrix_docs_topics_)
```

### Results visualization

You need to install [tmplot](https://github.com/maximtrp/tmplot) first.

```python
import tmplot as tmp
tmp.report(model=model, docs=texts)
```

![Report interface](images/topics_terms_plots.png)

## Tutorial

There is a [tutorial](https://bitermplus.readthedocs.io/en/latest/tutorial.html)
in documentation that covers the important steps of topic modeling (including
stability measures and results visualization).
