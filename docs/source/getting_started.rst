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

If this runs without error, you can now skip to the Obtain Data section.