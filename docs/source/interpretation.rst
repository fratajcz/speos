Model Interpretation
====================

Sometimes it can be very helpful to see which input features have been the most influential in a prediction. Not only does this give insight into which data is valuable to the task and which data can be omitted, it also increases confidence in predictions if the model interpretation is congruent with the domain.

To make use of the full range of model interpretation techniques that have been developed for neural networks, Speos uses the bridge between PyTorch Geometric and Captum, the largest model interpretation library for PyTorch.

We have encapsulated the necessary code into command line scripts which either provide interpretations for individual models or for entire ensembles using `Integrated Gradients <https://captum.ai/docs/extension/integrated_gradients>`_. All you have to do is provide the config file of the runs and the label or index of the genes for which interpretations should be generated:

.. note:: 

    This section is a work in progress and will likely change in the future.


Individual Models
-----------------

.. code-block:: console

    $ python speos/scripts/explanation_scripts/explanation_one_model.py -h
    usage: explanation_one_model.py [-h] [--config CONFIG] [--gene GENE]
                                [--index INDEX] [--threshold THRESHOLD]

    Run model interpretation for a selected gene and model

    optional arguments:
    -h, --help            show this help message and exit
    --config CONFIG, -c CONFIG
                            path to config of the run that should be examined
    --gene GENE, -g GENE  HGNC gene symbol which should be interpreted
    --index INDEX, -i INDEX
                            index of gene to examine
    --threshold THRESHOLD, -t THRESHOLD
                            minimum importance of nodes and edges required to be
                            plotted

So, by either specifying the HGNC gene symbol or the gene's ID in the graph, we can generate the interpretations for this genes prediction. The threshold flag is only relevant for plotting, in case there are too many nodes in the neighborhood.

In the end, it will produce a preliminary plot of the required genes neighborhood and place it in the config's :obj:`config.model.plot_dir`. Furthermore, more detailed lists of importance values for edges and node features are placed in :obj:`config.pp.save_dir`.


Whole Ensembles
---------------

To obtain the interpretation for a whole ensemble, each of the individual model interpreatations are gathered, minmax scaled to the interval [0,1] and then averaged. This means that in the end, an importance close to 1 means that this edge or input feauture is important for most of the models while an importance value close to 0 means that it is unimportant for most models.

If your input graph has only one edge type, use the following script:

.. code-block:: console

    $ python speos/scripts/explanation_scripts/explanation_ensemble_homogeneous.py -h
    usage: explanation_one_model.py [-h] [--config CONFIG] [--gene GENE]
                                [--index INDEX] [--threshold THRESHOLD]

    Run model interpretation for a selected gene and ensemble

    optional arguments:
    -h, --help            show this help message and exit
    --config CONFIG, -c CONFIG
                            path to config of the run that should be examined
    --gene GENE, -g GENE  HGNC gene symbol which should be interpreted
    --index INDEX, -i INDEX
                            index of gene to examine
    --threshold THRESHOLD, -t THRESHOLD
                            minimum importance of nodes and edges required to be
                            plotted
    --mincs MINCS, -m MINCS
                            minimal cs of candidates to examine
    --readonly, -r        if run should be readonly.
    --device DEVICE, -d DEVICE
                            The device on which the calculations should be ru on
                            (i.e. "cpu", or "cuda:0" etc.)

In addition to the aforementioned arguments, you can now also set the minimal Consensus Score (CS) as cutoff to select candidate genes that should be examined. Additionally, as this interpreation script includes heavier calculations than the other, it lets you specify if the run should be readonly (i.e. for testing) and on which device it should be run.

TODO: add script for multi-relatinal graphs.