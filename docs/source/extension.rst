Extending Speos
===============

The strength of Speos lies in its ability to be customized and extended. See the following sections on how to add your own network, additional GWAS data and new phenotype labels!

Adding A Network
----------------

To add an additional network to Speos, you can simply register it in ``extensions/adjacencies.json``. Se the following example how to do it.

Without any manipulation, ``extensions/adjacencies.json`` simply contains an empty list:

.. code-block:: json

    []

This is because Speos has defined its core networks elsewhere (in ``speos/adjacencies.json``).
To add a network, simply modify ``extensions/adjacencies.json`` as follows:


.. code-block:: json

    [{"name": "MyNetwork",
    "type": "mytype",
    "file_path": "path/to/the/file.tsv",
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