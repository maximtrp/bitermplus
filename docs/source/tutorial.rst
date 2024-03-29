Tutorial
========

Model fitting
-------------

Here is a simple example of model fitting.
It is supposed that you have already gone through the preprocessing
stage: cleaned, lemmatized or stemmed your documents, and removed stop words.

.. code-block:: python

    import bitermplus as btm
    import numpy as np
    import pandas as pd

    # Importing data
    df = pd.read_csv(
        'dataset/SearchSnippets.txt.gz', header=None, names=['texts'])
    texts = df['texts'].str.strip().tolist()

    # Vectorizing documents, obtaining full vocabulary and biterms
    # Internally, btm.get_words_freqs uses CountVectorizer from sklearn
    # You can pass any of its arguments to btm.get_words_freqs
    # For example, you can remove stop words:
    stop_words = ["word1", "word2", "word3"]
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts, stop_words=stop_words)
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    biterms = btm.get_biterms(docs_vec)

    # Initializing and running model
    model = btm.BTM(
        X, vocabulary, seed=12321, T=8, M=20, alpha=50/8, beta=0.01)
    model.fit(biterms, iterations=20)


Inference
---------

Now, we will calculate documents vs topics probability matrix (make an inference).

.. code-block:: python

    p_zd = model.transform(docs_vec)

If you need to make an inference on a new dataset, you should
vectorize it using your vocabulary from the training set:

.. code-block:: python

    new_docs_vec = btm.get_vectorized_docs(new_texts, vocabulary)
    p_zd = model.transform(new_docs_vec)


Calculating metrics
-------------------

To calculate perplexity, we must provide documents vs topics probability matrix
(``p_zd``) that we calculated at the previous step. 

.. code-block:: python

    perplexity = btm.perplexity(model.matrix_topics_words_, p_zd, X, 8)
    coherence = btm.coherence(model.matrix_topics_words_, X, M=20)
    # or
    perplexity = model.perplexity_
    coherence = model.coherence_


Visualizing results
-------------------

For results visualization, we will use `tmplot
<https://pypi.org/project/tmplot/>`_ package.

.. code-block:: python

    import tmplot as tmp

    # Run the interactive report interface
    tmp.report(model=model, docs=texts)

Filtering stable topics
-----------------------

Unsupervised topic models (such as LDA) are subject to topic instability [1]_
[2]_ [3]_. There is a special method in ``tmplot`` package for selecting stable
topics. It uses various distance metrics such as Kullback-Leibler divergence
(symmetric and non-symmetric), Hellinger distance, Jeffrey's divergence,
Jensen-Shannon divergence, Jaccard index, Bhattacharyya distance, Total
variation distance.

.. code-block:: python

    import pickle as pkl
    import tmplot as tmp
    import glob

    # Loading saved models
    models_files = sorted(glob.glob(r'results/model[0-9].pkl'))
    models = []
    for fn in models_files:
        file = open(fn, 'rb')
        models.append(pkl.load(file))
        file.close()

    # Choosing reference model
    np.random.seed(122334)
    reference_model = np.random.randint(1, 6)
    
    # Getting close topics
    close_topics, close_kl = tmp.get_closest_topics(
        models, method="sklb", ref=reference_model)

    # Getting stable topics
    stable_topics, stable_kl = tmp.get_stable_topics(
        close_topics, close_kl, ref=reference_model, thres=0.7)
    
    # Stable topics indices list
    print(stable_topics[:, reference_model])


Model loading and saving
------------------------

Support for model serializing with `pickle
<https://docs.python.org/3/library/pickle.html>`_ was implemented in v0.5.3.
Here is how you can save and load a model:

.. code-block:: python

    import pickle as pkl
    # Saving
    with open("model.pkl", "wb") as file:
        pkl.dump(model, file)

    # Loading
    with open("model.pkl", "rb") as file:
        model = pkl.load(file)


References
----------

.. [1] Koltcov, S., Koltsova, O., & Nikolenko, S. (2014, June).
   Latent dirichlet allocation: stability and applications to studies of
   user-generated content. In Proceedings of the 2014 ACM conference on Web
   science (pp. 161-165).

.. [2] Mantyla, M. V., Claes, M., & Farooq, U. (2018, October).
   Measuring LDA topic stability from clusters of replicated runs. In
   Proceedings of the 12th ACM/IEEE international symposium on empirical
   software engineering and measurement (pp. 1-4).

.. [3] Greene, D., O’Callaghan, D., & Cunningham, P. (2014, September). How many
   topics? stability analysis for topic models. In Joint European conference on
   machine learning and knowledge discovery in databases (pp. 498-513). Springer,
   Berlin, Heidelberg.
