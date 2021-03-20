Tutorial
========

Model fitting
-------------

Here is a simple example of package usage:

.. code-block:: python

    import bitermplus as btm
    import numpy as np
    from gzip import open as gzip_open

    # Importing and vectorizing text data
    with gzip_open('dataset/SearchSnippets.txt.gz', 'rb') as file:
        texts = file.readlines()

    # Vectorizing documents, obtaining full vocabulary and biterms
    X, vocab = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(X)
    biterms = btm.get_biterms(X)

    # Initializing and running model
    model = btm.BTM(X, vocab, T=8, W=vocab.size, M=20, alpha=50/8, beta=0.01)
    model.fit(biterms, iterations=20)
    p_zd = model.transform(docs_vec)

    # Calculating metrics
    perplexity = btm.perplexity(model.matrix_words_topics_, p_zd, X, 8)
    coherence = btm.coherence(model.matrix_words_topics_, X, M=20)
    # or
    perplexity = model.perplexity_
    coherence = model.coherence_

Model loading and saving
------------------------

Support for model serializing with `pickle <https://docs.python.org/3/library/pickle.html>`_ was implemented in v0.5.3. Here is how you can save and load a model:

.. code-block:: python

    import pickle as pkl
    # Saving
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)

    # Loading
    with open("model.pkl", "rb") as file:
        model = pkl.load(file)
