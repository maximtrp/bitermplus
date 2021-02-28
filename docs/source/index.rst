bitermplus
==========

This package implements `Biterm topic
model <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf>`_
for short texts introduced by Xiaohui Yan, Jiafeng Guo, Yanyan Lan, and Xueqi
Cheng. It is based on `biterm <https://github.com/markoarnauto/biterm>`_ package
by `@markoarnauto <https://github.com/markoarnauto>`_. Unfortunately, *biterm*
package is not maintained anymore.

*Bitermplus* is a fixed and optimized successor. Pure Python version of ``BTM``
class was removed. Class ``oBTM`` was strongly optimized using typed memoryviews
in Cython and now replaces ``BTM`` class.

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   Installation <install>
   Tutorial <tutorial>

.. toctree::
   :maxdepth: 2
   :caption: API
   :hidden:

   bitermplus <bitermplus>
   bitermplus.metrics <bitermplus.metrics>
   bitermplus.plot <bitermplus.plot>
   bitermplus.util <bitermplus.util>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
