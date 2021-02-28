Tutorial
========

Here is a simple example of package usage:

.. code-block:: python

    import bitermplus as btm
    import numpy as np
    from gzip import open as gzip_open

    # Importing and vectorizing text data
    with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
        texts = file.readlines()

    # Vectorizing documents, obtaining full vocabulary and biterms
    X, vocab = btm.util.get_vectorized_docs(texts)
    biterms = btm.util.get_biterms(X)

    # Initializing and running model
    model = btm.BTM(X, T=8, W=vocab.size, M=20, alpha=50/8, beta=0.01, L=0.5)
    model.fit(biterms, iterations=10)
    P_zd = model.transform(biterms)

    # Calculating metrics
    perplexity = btm.metrics.perplexity(model.phi_, P_zd, X, 8)
    coherence = btm.metrics.coherence(model.phi_, X, M=20)
    # or
    perplexity = model.perplexity_
    coherence = model.coherence_
