Installation
------------

Linux and Windows
~~~~~~~~~~~~~~~~~

Install *bitermplus* directly from PyPI:

.. code-block:: bash

    pip install bitermplus

For the latest development version:

.. code-block:: bash

    pip install git+https://github.com/maximtrp/bitermplus.git

Mac OS
~~~~~~

First, you need to install XCode CLT and `Homebrew <https://brew.sh>`_.
Then, install ``libomp`` using ``brew``:

.. code-block:: bash

    xcode-select --install
    brew install libomp
    pip3 install bitermplus

Requirements
~~~~~~~~~~~~

* cython
* numpy
* pandas
* scipy
* scikit-learn
* tqdm
