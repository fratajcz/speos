Extending Speos
===============

The strength of Speos lies in its ability to be customized and extended. See the following sections on how to add your own network, additional GWAS data and new phenotype labels!

Adding A Network
----------------

Say you have an adjacency that you want to add to Speos to run some experiments on. For the sake of simplicity, lets say your edgelist contains only one edge and looks like this:

.. code-block:: text

    SymbolA SymbolB
    MTOR    IL1B

It describes a fictional connection between the MTOR and the IL1B gene using HGNC gene identifier. You now save this edgelist to ``data/myadjacency/edgelist.tsv``  
Now, the next step is to tell Speos how to use your new adjacency. To add an additional network to Speos, you can simply register it in ``extensions/adjacencies.json``. Se the following example how to do it:

Without any manipulation, ``extensions/adjacencies.json`` simply contains an empty list:

.. code-block:: json

    []

This is because Speos has defined its core networks elsewhere (in ``speos/adjacencies.json``).
To add a network, simply modify ``extensions/adjacencies.json`` as follows:


.. code-block:: json

    [{"name": "MyNetwork",
    "type": "ppi",
    "file_path": "data/myadjacency/edgelist.tsv",
    "source": "SymbolA",
    "target": "SymbolB",
    "sep": "\t",
    "symbol": "hgnc",
    "weight": "None",
    "directed": false}]

:obj:`"name"`: This key specifies the name of the network, how it should be called in logging and plotting and with which name it should be matched during the search.

:obj:`"type"` :This key specifiees the network's type and allows type-specific multi-network runs. Currently in use are "ppi", "grn", "evo" and "metabolic". If you set it to "ppi", you can blend it in with the other PPIs in a multi-network run. If you want to use you network in isolation, then this key is not important. If you want to add multiple networks, you can create your own type (i.e. "mytype") and use this to cluster your netowrks.

:obj:`"file_path"`: Here you specify the path to the edgelist file starting in the speos main directory.

:obj:`"source"` and :obj:`"target"`: Tese keys specify the column headers where the source and target nodes are specified for every edge.

:obj:`"sep"`: This key specifies the column seperator of the file.

:obj:`"symbol"`: This key specifies which type of symbol is used to identify the gene, use either "hgnc", "entrez" or "ensembl".

:obj:`"weight"`: This key specifies if there is a column that contains edge weights. "None" means there are no edge weights (all have weight 1), otherwise specify the column header here. (not implemented yet)

:obj:`"directed"`: This key contains a boolean (false/true) and tells Speos if the edges are directed or undirected.

Using your Network
------------------

To use the network that we just have added to ``extensions/adjacencies.json``, you can simply set according values in a config file and try to run a training run using that config file.

Select your network in the config ``my_config.yaml``:

.. code-block:: text

    name: test_adjacency

    input:
        adjacency: MyNetwork
        adjacency_field: name

This config looks at the :obj:`"name"` tags of all available adjacencies and selects thos that match the value defined in :obj:`"adjacency"`.

We save this config and risk a testrun:

.. code-block:: console

    $ python training.py -c my_config.yaml
    test_adjacency 2022-08-29 16:43:17,430 [INFO] speos.experiment: Starting run test_adjacency
    test_adjacency 2022-08-29 16:43:17,430 [INFO] speos.experiment: Cuda is available: True
    test_adjacency 2022-08-29 16:43:17,430 [INFO] speos.experiment: Using device(s): ['cuda:0']
    Processing...
    Done!
    test_adjacency 2022-08-29 16:43:17,432 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['MyNetwork']
    test_adjacency 2022-08-29 16:43:17,432 [INFO] speos.preprocessing.preprocessor: Using 8 mappings with ground truth ./data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    test_adjacency 2022-08-29 16:43:35,445 [INFO] speos.preprocessing.preprocessor: Name: 
    Type: MultiDiGraph
    Number of nodes: 16852
    Number of edges: 2
    Average in degree:   0.0001
    Average out degree:   0.0001
    test_adjacency 2022-08-29 16:53:37,108 [INFO] speos.datasets: Data(x=[16852, 96], edge_index=[2, 2], y=[16852], train_mask=[16852], test_mask=[16852], val_mask=[16852])

We can see that the network has been processed by looking at the logging output in the terminal. Our graph has 16852 nodes but only 2 edges!

Using your Network together with others
---------------------------------------

We can use the :obj:`"type"` keyword in the description of the network to trigger a multi-network run. The :obj:`"type"` of our little netork is set to :obj:`"ppi"`, so we can blend it with other PPIs by using the folling config:

Adapt ``my_config.yaml`` to:

.. code-block:: text

    name: test_adjacency

    input:
        adjacency: ppi
        adjacency_field: type

and run it, which results in a different Output:

.. code-block:: console

    $ python training.py -c my_config.yaml
    test_adjacency 2022-08-29 16:59:23,197 [INFO] speos.experiment: Starting run test_adjacency
    test_adjacency 2022-08-29 16:59:23,197 [INFO] speos.experiment: Cuda is available: True
    test_adjacency 2022-08-29 16:59:23,198 [INFO] speos.experiment: Using device(s): ['cuda:0']
    Processing...
    Done!
    test_adjacency 2022-08-29 16:59:23,202 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30HCT116', 'BioPlex30293T', 'HuRI', 'IntActPA', 'IntActDirect', 'MyNetwork']
    test_adjacency 2022-08-29 16:59:23,202 [INFO] speos.preprocessing.preprocessor: Using 8 mappings with ground truth ./data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    test_adjacency 2022-08-29 17:00:22,636 [INFO] speos.preprocessing.preprocessor: Name: 
    Type: MultiDiGraph
    Number of nodes: 16852
    Number of edges: 613054
    Average in degree:  36.3787
    Average out degree:  36.3787
    test_adjacency 2022-08-29 17:00:24,158 [INFO] speos.datasets: HeteroData(
    x=[16852, 96],
    y=[16852],
    train_mask=[16852],
    test_mask=[16852],
    val_mask=[16852],
    gene={ x=[16852, 96] },
    (gene, BioPlex30HCT116, gene)={ edge_index=[2, 97270] },
    (gene, BioPlex30293T, gene)={ edge_index=[2, 158962] },
    (gene, HuRI, gene)={ edge_index=[2, 78586] },
    (gene, IntActPA, gene)={ edge_index=[2, 205718] },
    (gene, IntActDirect, gene)={ edge_index=[2, 14274] },
    (gene, MyNetwork, gene)={ edge_index=[2, 2] }
    )

Now we see that we use multiple adjacencies, including MyNetwork!