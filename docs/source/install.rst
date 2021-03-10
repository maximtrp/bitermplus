Setup
-----

Linux and Windows
~~~~~~~~~~~~~~~~~

There should be no issues with installing *bitermplus* under these OSes. You can install the package directly from PyPi.

.. code-block:: bash

    pip install bitermplus

Or from this repo:

.. code-block:: bash

    pip install git+https://github.com/maximtrp/bitermplus.git

Mac OS
~~~~~~

Currently, there is an issue in package compiling related to Clang and ``openmp`` flag.
I recommend using GNU GCC compiler as a temporary workaround.
First, you need to install XCode CLT and `Homebrew <https://brew.sh>`_.
Then, install GCC using ``brew`` and export ``CC`` variable with the path to GCC compiler.

.. code-block:: bash

    xcode-select --install
    brew install gcc
    export CC=/usr/local/bin/gcc-10
    pip3 install bitermplus

Requirements
~~~~~~~~~~~~

* Cython
* NumPy
* Pandas
* SciPy
* Scikit-learn
* pyLDAvis (optional)