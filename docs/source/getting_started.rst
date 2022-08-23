Getting Started
===============

Speos is fully implemented in python with some sidearms in R. To use Speos, either as a full end-to-end framework or just parts of it for visualization or preprocessing, 
follow the upcoming sections to get started.

Quickstart: Docker
------------------

We provide a docker image that is pre-packaged with Speos and all dependencies. You can use it to start experimenting and check if Speos has what you want.
However, since CUDA is highly hardware-dependent, we only provide a docker image for the CPU implementation (for now). 
If you want to run Speos on the GPU you can modify the dockerfile to your needs or continue with the Installation section.

To build the image, run the following commands (while in the speos main dir):

.. code-block:: console

    $ git clone https://github.com/fratajcz/speos.git
    $ cd speos
    $ docker build -t speos . -f docker/pyg_cpu


Note that this takes a while. Then you can run the image with the following command and test that it compiled correctly:

.. code-block:: console

    $ docker run -ti speos
    user@a2c4c28ccc79:/app$ python
    Python 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.

>>>import speos

If this runs without error, you can now skip to the :ref:`Obtain Data` section.

Installation
------------

To Install Speos on your machine, first clone it with its submodules from github:

.. code-block:: console

    $ git clone --recurse-submodules https://github.com/fratajcz/speos.git
    $ cd speos

Next, you might want to create a new conda env.

.. code-block:: console

    $ conda create -n speos
    $ conda activate speos

Speos is built on Pytorch Geometric which in turn is built on Pytorch. To make sure Speos runs correctly, you must first install Pytorch and Pytorch geometric.
Note that these following lines install the CPU version of the packages. Since the GPU version depend very much on the hardware configuration of the system, 
please see X and Y on how to install Pytorch and Pytorch Geoemtric GPU versions for your hardware stack. 

.. code-block:: console

    $ conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts -y
    $ conda install pyg -c pyg -y

Then, install the remaining requirements with pip:

.. code-block:: console

    $ python3 -m pip install -r requirements.yaml

And finally install speos (make sure that you are in the main repo of speos):

.. code-block:: console

    $ pip install .

You can now test if it installed correctly by typing:

.. code-block:: console

    $ python
    Python 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>import speos

if any of those steps produced an error, please do not hesitate and open an issue in Github.

Obtain Data
-----------

You might have notived that :obj:`speos/data` is empty. This is because Speos is built on too much data to store it in Github. To obtain the data, run the following command in the Speos main dir:

.. code-block:: console

    $ ./download_data.sh

Test it
-------

If everything has gone right, the following command should start preprocessing data and train a model:

.. code-block:: console

  $ python training.py

Congratulations, you can now proceed to the :ref:`api<API>`` section to see how you can customize Speos to your needs!