Sklearn-style API
=================

The bitermplus package now includes a sklearn-compatible API that makes topic modeling much easier and more intuitive. The :class:`BTMClassifier` class provides a familiar interface for scikit-learn users and integrates seamlessly with ML pipelines.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

The new API reduces complex topic modeling workflows to just a few lines:

.. code-block:: python

   import bitermplus as btm

   # Sample documents
   texts = [
       "machine learning algorithms are powerful",
       "deep learning neural networks process data",
       "natural language processing understands text",
       "artificial intelligence transforms industries"
   ]

   # Create and fit model (one step!)
   model = btm.BTMClassifier(n_topics=2, random_state=42)
   model.fit(texts)

   # Get topic distributions
   doc_topics = model.transform(texts)
   print(f"Document-topic matrix shape: {doc_topics.shape}")

   # Interpret topics
   topic_words = model.get_topic_words(n_words=5)
   for topic_id, words in topic_words.items():
       print(f"Topic {topic_id}: {', '.join(words)}")

API Comparison
--------------

**Traditional API** (complex, multi-step)::

   # Multiple manual preprocessing steps
   X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
   docs_vec = btm.get_vectorized_docs(texts, vocabulary)
   biterms = btm.get_biterms(docs_vec)

   # Model creation and fitting
   model = btm.BTM(X, vocabulary, seed=42, T=3, M=20, alpha=50/3, beta=0.01)
   model.fit(biterms, iterations=100)

   # Inference
   p_zd = model.transform(docs_vec)

**New Sklearn API** (simple, one-liner)::

   # Everything in one step!
   model = btm.BTMClassifier(n_topics=3, random_state=42)
   doc_topics = model.fit_transform(texts)

BTMClassifier Class
-------------------

.. currentmodule:: bitermplus

.. autoclass:: BTMClassifier
   :members:
   :inherited-members:
   :show-inheritance:

   .. automethod:: __init__

Core Methods
~~~~~~~~~~~~

The :class:`BTMClassifier` follows the sklearn estimator interface:

**fit(X, y=None)**
   Train the BTM model on documents.

**transform(X, infer_type='sum_b')**
   Transform documents to topic probability distributions.

**fit_transform(X, y=None, infer_type='sum_b')**
   Fit model and transform documents in one step.

**score(X, y=None)**
   Return mean coherence score across topics.

Parameters
~~~~~~~~~~

**n_topics** : int, default=8
   Number of topics to extract.

**alpha** : float, default=None
   Dirichlet prior for topic distribution. If None, uses 50/n_topics.

**beta** : float, default=0.01
   Dirichlet prior for word distribution.

**max_iter** : int, default=600
   Maximum iterations for model training.

**random_state** : int, default=None
   Random seed for reproducible results.

**window_size** : int, default=15
   Window size for biterm generation.

**vectorizer_params** : dict, default=None
   Parameters for the internal CountVectorizer.

Topic Analysis Methods
~~~~~~~~~~~~~~~~~~~~~~

**get_topic_words(topic_id=None, n_words=10)**
   Get top words for topics. Returns list for single topic or dict for all topics.

**get_document_topics(X, threshold=0.1)**
   Get dominant topics for documents above probability threshold.

Properties
~~~~~~~~~~

**coherence_** : np.ndarray
   Topic coherence scores.

**perplexity_** : float
   Model perplexity (requires transform to be called first).

**topic_word_matrix_** : np.ndarray
   Topics × words probability matrix.

**vocabulary_** : np.ndarray
   Learned vocabulary.

**n_features_in_** : int
   Number of features (vocabulary size).

Sklearn Integration
-------------------

Cross-validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   model = btm.BTMClassifier(n_topics=5, random_state=42)
   scores = cross_val_score(model, texts, cv=3)
   print(f"Mean coherence: {scores.mean():.3f}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import FunctionTransformer

   def preprocess_text(texts):
       return [text.lower().replace(',', '') for text in texts]

   pipeline = Pipeline([
       ('preprocess', FunctionTransformer(preprocess_text)),
       ('btm', btm.BTMClassifier(n_topics=3, random_state=42))
   ])

   doc_topics = pipeline.fit_transform(texts)

Grid Search
~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'n_topics': [3, 5, 8],
       'alpha': [0.1, 0.5, 1.0],
       'max_iter': [100, 300]
   }

   grid_search = GridSearchCV(
       btm.BTMClassifier(random_state=42),
       param_grid,
       cv=3,
       scoring=None  # Uses model's score method
   )

   grid_search.fit(texts)
   best_model = grid_search.best_estimator_

Advanced Usage
--------------

Custom Preprocessing
~~~~~~~~~~~~~~~~~~~~

Control text preprocessing with ``vectorizer_params``:

.. code-block:: python

   custom_params = {
       'min_df': 2,           # Minimum document frequency
       'max_df': 0.8,         # Maximum document frequency
       'stop_words': 'english',  # Remove English stop words
       'lowercase': True,     # Convert to lowercase
       'token_pattern': r'\b[a-zA-Z]{3,}\b'  # Only words 3+ chars
   }

   model = btm.BTMClassifier(
       n_topics=5,
       vectorizer_params=custom_params
   )

Inference Types
~~~~~~~~~~~~~~~

Choose different inference methods:

.. code-block:: python

   model = btm.BTMClassifier(n_topics=5)
   model.fit(texts)

   # Different inference types
   topics_sum_b = model.transform(new_texts, infer_type='sum_b')  # Default
   topics_sum_w = model.transform(new_texts, infer_type='sum_w')  # Word-based
   topics_mix = model.transform(new_texts, infer_type='mix')      # Mixed

Model Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   model = btm.BTMClassifier(n_topics=5, random_state=42)
   model.fit(texts)

   # Coherence per topic
   coherence_scores = model.coherence_
   print(f"Topic coherence: {coherence_scores}")

   # Overall model quality
   mean_coherence = model.score(texts)
   print(f"Mean coherence: {mean_coherence:.3f}")

   # Perplexity (lower is better)
   model.transform(texts)  # Required for perplexity calculation
   perplexity = model.perplexity_
   print(f"Perplexity: {perplexity:.3f}")

Working with Pandas
~~~~~~~~~~~~~~~~~~~

The API works seamlessly with pandas DataFrames:

.. code-block:: python

   import pandas as pd

   df = pd.DataFrame({'text': texts, 'category': ['ML', 'DL', 'NLP', 'AI']})

   model = btm.BTMClassifier(n_topics=3)
   doc_topics = model.fit_transform(df['text'])

   # Add topic predictions to DataFrame
   df['dominant_topic'] = doc_topics.argmax(axis=1)
   df['topic_confidence'] = doc_topics.max(axis=1)

Tips and Best Practices
-----------------------

Parameter Selection
~~~~~~~~~~~~~~~~~~~

- **n_topics**: Start with 5-10 topics for small datasets, 10-50 for larger ones
- **alpha**: Higher values (1.0+) create more evenly distributed topics
- **beta**: Keep small (0.01-0.1) for focused topics
- **max_iter**: 100-200 usually sufficient for convergence

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

- Use ``random_state`` for reproducible results
- Set ``max_iter`` lower for faster experimentation
- Adjust ``vectorizer_params`` to control vocabulary size
- For large datasets, consider increasing ``min_df`` to reduce vocabulary

Topic Quality
~~~~~~~~~~~~~

- Check coherence scores - higher is generally better
- Examine top words per topic for interpretability
- Use ``get_document_topics()`` to see topic assignments
- Compare different ``n_topics`` values using coherence

Common Issues
-------------

**Import Errors**
   Make sure Cython extensions are built: ``python setup.py build_ext --inplace``

**Empty Topics**
   Reduce ``n_topics`` or adjust ``vectorizer_params`` (lower ``min_df``)

**Poor Topic Quality**
   Try different ``alpha``/``beta`` values or increase ``max_iter``

**Memory Issues**
   Increase ``min_df`` to reduce vocabulary size for large datasets

Migration Guide
---------------

Converting from Original API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Old code:**

.. code-block:: python

   # Original bitermplus API
   X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
   docs_vec = btm.get_vectorized_docs(texts, vocabulary)
   biterms = btm.get_biterms(docs_vec)

   model = btm.BTM(X, vocabulary, seed=42, T=8, M=20, alpha=50/8, beta=0.01)
   model.fit(biterms, iterations=600)
   p_zd = model.transform(docs_vec)

**New code:**

.. code-block:: python

   # New sklearn-style API
   model = btm.BTMClassifier(
       n_topics=8,
       random_state=42,
       coherence_window=20,
       alpha=50/8,
       beta=0.01,
       max_iter=600
   )
   p_zd = model.fit_transform(texts)

The new API handles all preprocessing automatically while providing the same underlying functionality with much simpler usage.