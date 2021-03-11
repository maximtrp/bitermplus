Setup
-----

Linux and Windows
~~~~~~~~~~~~~~~~~

There should be no issues with installing *bitermplus* under these OSes.
You can install the package directly from PyPi.

.. code-block:: bash

    pip install bitermplus

Or from this repo:

.. code-block:: bash

    pip install git+https://github.com/maximtrp/bitermplus.git

Mac OS
~~~~~~

First, you need to install XCode CLT and `Homebrew <https://brew.sh>`_.
Then, install `libomp` using `brew`:

.. code-block:: bash

    xcode-select --install
    brew install libomp
    pip3 install bitermplus

Requirements
~~~~~~~~~~~~

* Cython
* NumPy
* Pandas
* SciPy
* Scikit-learn
* pyLDAvis (optional)