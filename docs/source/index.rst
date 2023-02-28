.. Speos documentation master file, created by
   sphinx-quickstart on Wed Aug 17 13:27:31 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: https://raw.githubusercontent.com/fratajcz/speos/master/img/speos_space_11_1080.png
  :width: 640
  :alt: Speos Banner

Welcome to Speos's documentation!
=================================

Speos is a framework built on `PyG <https://pytorch-geometric.readthedocs.io>`_ (PyTorch Geometric) and `PyTorch <https://pytorch.org/>`_ to easily train and evaluate Graph Neural Networks (GNNs) and other machine learning models for gene classification.

It handles the integration of various types of genomic data, as well as the complete training and evaluation process. For the motivation behind the framework you can check our `preprint <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1.full.pdf>`_ with exciting applications and showcases.

Speos is designed as a low code platform, meaning that users will have to write little to no code to access the full bandwidth of state-of-the-art methods and datasets. It is possible to extend Speos to your needs, which might come with a minimum of implementation effort, as detailed in the later chapters of this documentation.

For now, you can go through the chapters of this documenation one by one and learn how to install, use, adapt and extend Speos to power your experiments!

.. note::

   If you are facing any issues or have questions, do not hesitate to file a bug report or feature request via `GitHub Issues <https://github.com/fratajcz/speos/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: Step By Step
   
   introduction
   getting_started
   api
   configuration
   extension
   benchmarking
   validation
   ensemble
   interpretation

.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   
   inputs
   postprocessing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
