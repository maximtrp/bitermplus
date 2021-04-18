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
    X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
    docs_vec = btm.get_vectorized_docs(texts, vocabulary)
    biterms = btm.get_biterms(docs_vec)

    # Initializing and running model
    model = btm.BTM(
        X, vocabulary, seed=12321, T=8, W=vocabulary.size, M=20, alpha=50/8, beta=0.01)
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

For results visualization, we will use `pyLDAvis
<https://pypi.org/project/pyLDAvis/>`_ package.

.. code-block:: python

    # Calculate terms frequency
    tf = np.array(X.sum(axis=0)).ravel()

    # Calculate vectorized documents lengths
    docs_lens = list(map(len, docs_vec))

    # Prepare results for visualization
    vis = btm.vis_prepare_model(
        model_ref.matrix_topics_words_,
        dtd,
        docs_lens,
        model_ref.vocabulary_,
        tf
    )
    # Enable Jupyter notebook support
    plv.enable_notebook()

    # Finally, display the results
    plv.display(vis)


Filtering stable topics
-----------------------

Unsupervised topic models (such as LDA) are subject to topic instability [1]_ [2]_ [3]_.
There are several methods in ``bitermplus`` package for selecting stable topics:
Kullback-Leibler divergence, Hellinger distance, Jeffrey's divergence, Jensen-Shannon divergence,
Jaccard index, Bhattacharyya distance.

.. code-block:: python

    import pickle as pkl
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
    close_topics, close_kl = btm.get_closest_topics(
        *list(map(lambda x: x.matrix_topics_words_, models)),
        method="sklb", ref=reference_model)

    # Getting stable topics
    stable_topics, stable_kl = btm.get_stable_topics(
        close_topics, close_kl, ref=reference_model, thres=0.7)
    
    # Stable topics indices list
    print(stable_topics[:, reference_model])


Model loading and saving
------------------------

Support for model serializing with `pickle <https://docs.python.org/3/library/pickle.html>`_ was implemented in v0.5.3.
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