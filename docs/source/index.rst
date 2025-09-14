bitermplus
==========

**Bitermplus** implements the `Biterm Topic Model (BTM)
<https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf>`_
for short text analysis, developed by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi
Cheng. This is a high-performance Cython implementation of the original `BTM
<https://github.com/xiaohuiyan/BTM>`_ with OpenMP parallelization. The package includes
comprehensive evaluation metrics including *perplexity* and *semantic coherence*.

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   Installation <install>
   Sklearn API <sklearn_api>
   Tutorial <tutorial>
   Benchmarks <benchmarks>

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   Model <bitermplus>
   Metrics <bitermplus.metrics>
   Utility functions <bitermplus.util>
