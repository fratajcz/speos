Getting Started
===============

Speos is fully implemented in python with some sidearms in R. To use Speos, either as a full end-to-end framework or just parts of it for visualization or preprocessing, 
follow the upcoming sections to get started.

Docker
------

We provide a dockerfile that compiles an image that is pre-packaged with Speos and all dependencies. You can use it to start experimenting and check if Speos has what you want.
However, since CUDA is highly hardware-dependent, we only provide a docker image for the CPU implementation (for now). 
If you want to run Speos on the GPU you can modify the dockerfile to your needs or continue with the :ref:`Installation` section.

.. note::

    We are unable to share pre-compiled docker images due to the number of possible OS and hardware combinations of user systems and the large size of the images (> 5GB for CPU version, >18GB for GPU version) for each of those combinations.
    Be patient if the compilation of dependencies in docker takes a while, in our experience it can take up to an hour but it normally finishes just fine.


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

Installing Speos on your machine is straight-forward and can be accomplished within 15 minutes given a familiarity with python package manager tools like :obj:`pip` and :obj:`conda`.

.. note::

    These installation instructions have been tested under Ubuntu 20.04 and Rocky Linux 8.7. Other operating systems (i.e. MacOS, Windows) might require different package versions, especially for PyTorch and PyTorch Geometric.
    Although choosing newer versions of packages should generally work, we are unable to test each combination of package versions a priori. If you have questions about the installation of other package versions, do not hesitate and open an `Issue on Github <https://github.com/fratajcz/speos/issues>`_

To Install Speos on your machine, first clone it with its submodules from github:

.. code-block:: console

    $ git clone --recurse-submodules https://github.com/fratajcz/speos.git
    $ cd speos

Next, you might want to create a new conda env.

.. code-block:: console

    $ conda create -y --name speos python=3.7
    $ conda activate speos

Speos is built on Pytorch Geometric which in turn is built on Pytorch. To make sure Speos runs correctly, you must first install Pytorch and Pytorch geometric.
Note that these following lines install the CPU version of the packages. Since the GPU version depend very much on the hardware configuration of the system, 
please see how to install `Pytorch <https://pytorch.org/get-started/locally/>`_ and `Pytorch Geoemtric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ GPU versions for your hardware stack. 
If your machine/cluster has GPU capabilities, it is highly recommended to use the respective CUDA versions, as this will speed up the training a lot!

.. code-block:: console

    $ conda install pytorch torchvision torchaudio cpuonly -c pytorch-lts -y
    $ conda install pyg==2.0.4 -c pyg -y

.. note::

    If the last line (installing pyg with conda) did not work, i.e. if you work on a cluster with limited dependencies, try 

    .. code-block:: console

        $ pip install torch-scatter torch-sparse==0.6.12 torch-cluster torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-{1}+{2}.html

    instead. Replace the placeholder ${1}$ with the pytorch version you just installed (i.e. 1.8.0, 1.13.1 etc) and ${2}$ with the CUDA version (i.e. cpu for CPU only or cu116 for CUDA 11.6 etc.)

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

if any of those steps produced an error, please do not hesitate and open an issue on `Github <https://github.com/fratajcz/speos/issues>`_

Obtain Data
-----------

You might have noticed that :obj:`speos/data` is mostly empty. This is because Speos is built on too much data to store it on Github. To obtain the data in one step (~800MB), run the following command in the Speos main dir:

.. code-block:: console

    $ ./download_data.sh

If you are only interested in parts of the data, i.e. for a different project, check the individual subdirectories of :obj:`speos/data` and you will find download scripts for most programmatically accessible files that will download the file directly from the source repository.

Test it
-------

If everything has gone right, the following command should start preprocessing data and train a model:

.. code-block:: console

  $ python training.py

Now, you should see an output that resembles the following, just with :obj:`cde005` replaced with another random 6-character hash:

.. console::
    :linenos:

    cde005 2023-04-05 11:18:59,759 [INFO] speos.experiment: Starting run cde005
    cde005 2023-04-05 11:18:59,764 [INFO] speos.experiment: Cuda is available: False
    cde005 2023-04-05 11:18:59,764 [INFO] speos.experiment: CUDA set to auto, no CUDA device detected, setting to CPU
    cde005 2023-04-05 11:18:59,764 [INFO] speos.experiment: Using device(s): ['cpu']
    cde005 2023-04-05 11:18:59,779 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30293T']
    cde005 2023-04-05 11:18:59,782 [INFO] speos.preprocessing.preprocessor: Using 9 mappings with ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    Processing...
    cde005 2023-04-05 11:22:06,660 [INFO] speos.preprocessing.preprocessor: MultiDiGraph with 17024 nodes and 160962 edges
    Done!
    cde005 2023-04-05 11:22:09,140 [INFO] speos.preprocessing.preprocessor: Number of positives in ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed: 525
    cde005 2023-04-05 11:22:13,555 [INFO] speos.preprocessing.datasets: Loading Processed Data from ./data/processed/cde005.pt
    cde005 2023-04-05 11:22:14,030 [INFO] speos.experiment: GeneNetwork(
    (pre_mp): Sequential(
        (0): Linear(93, 50, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(50, 50, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(50, 50, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(50, 50, bias=True)
        (7): ELU(alpha=1.0)
        (8): Linear(50, 50, bias=True)
        (9): ELU(alpha=1.0)
        (10): Linear(50, 50, bias=True)
        (11): ELU(alpha=1.0)
    )
    (post_mp): Sequential(
        (0): Linear(50, 50, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(50, 50, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(50, 50, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(50, 50, bias=True)
        (7): ELU(alpha=1.0)
        (8): Linear(50, 50, bias=True)
        (9): ELU(alpha=1.0)
        (10): Linear(50, 25, bias=True)
        (11): ELU(alpha=1.0)
        (12): Linear(25, 1, bias=True)
    )
    (mp): Sequential(
        (0): GCNConv(50, 50)
        (1): ELU(alpha=1.0)
        (2): InstanceNorm(50)
        (3): GCNConv(50, 50)
        (4): ELU(alpha=1.0)
        (5): InstanceNorm(50)
    )
    )
    cde005 2023-04-05 11:22:14,365 [INFO] speos.preprocessing.datasets: Data(x=[17024, 93], edge_index=[2, 160962], y=[17024], train_mask=[17024], test_mask=[17024], val_mask=[17024])
    cde005 2023-04-05 11:22:14,452 [INFO] speos.experiment: Cuda is available: False
    cde005 2023-04-05 11:22:14,453 [INFO] speos.experiment: CUDA set to auto, no CUDA device detected, setting to CPU
    cde005 2023-04-05 11:22:14,520 [INFO] speos.experiment: Created new ResultsHandler pointing to ./results/cde005.h5
    cde005 2023-04-05 11:22:14,529 [INFO] speos.experiment: Received data with 472 train positives, 14849 train negatives, 27 val positives, 825 val negatives, 26 test positives and 825 test negatives
    cde005 2023-04-05 11:22:19,084 [INFO] speos.experiment: Writing TensoBoard data to ./inference/cde005
    cde005 2023-04-05 11:22:19,151 [INFO] speos.experiment: Writing TensoBoard data to ./runs/cde005
    cde005 2023-04-05 11:22:19,172 [INFO] speos.experiment: Epoch 0
    cde005 2023-04-05 11:22:47,629 [INFO] speos.experiment: Training Loss: 0.1978577714313068
    cde005 2023-04-05 11:22:58,102 [INFO] speos.experiment: Eval Loss: 0.19540250487193395, Accuracy: 0.9683098591549296, Recall: 0.0, Precision: 0.0, AUROC: 0.644354657687991, AUPRC: 0.05257679224153335, F1: 0.0, MRR: 0.0006422621747594641, MR: 5829.185185185185, Target: val
    cde005 2023-04-05 11:22:58,135 [INFO] speos.experiment: Epoch 1
    ...

With more training epochs to follow. As this run was started only to test the installation, feel free to cancel the run with a KeybordInterrupt (usually Ctrl+C).

Congratulations, you can now proceed to the API section to see how you can customize Speos to your needs!
