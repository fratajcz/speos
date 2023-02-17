Inputs
======

Speos automatically integrates various types of inputs, namely GWAS summary statistics, gene expression data and different biological networks.
To allow an extensible and easy integration of new data sources, please read the following documentation.


Quickstart
----------

Speos has an :obj:`InputHandler` class that requires only a config file and returns the fully equipped preprocessor. Use this class if all you want is the data for a given run:

.. autoclass:: speos.preprocessing.handler.InputHandler
    :members:
    :inherited-members:


Preprocessor
------------

The Preprocessor is a rather extensive class that strings together all preprocessing operations, such as reading files, building the graph from edgelists and normalizing input features.
It has a few useful functions for users, such as :obj:`get_data()`, :obj:`get_graph()` or :obj:`get_feature_names()`

.. autoclass:: speos.preprocessing.preprocessor.PreProcessor
    :members:
    :inherited-members:


GWAS Data
---------

The mapping of phenotypes to appropriate GWAS traits is done by the :obj:`speos.preprocessing.mappers.GWASMapper` :

.. autoclass:: speos.preprocessing.mappers.GWASMapper
    :members:
    :inherited-members:


Biological Networks
-------------------

The mapping of networks and filtering by their properties is done by the :obj:`speos.preprocessing.mappers.AdjacencyMapper` :

.. autoclass:: speos.preprocessing.mappers.AdjacencyMapper
    :members:
    :inherited-members:

For example:

>>> from speos.preprocessing.mappers import AdjacencyMapper
>>> mapper = AdjacencyMapper()     # initialize with default
>>>
>>> # get BioPlex 3.0 293T
>>> mapper.get_mappings(tags="BioPlex 3.0 293T", fields="name")
[{'name': 'BioPlex30293T', 'type': 'ppi', 'file_path': 'data/ppi/BioPlex_293T_Network_10K_Dec_2019.tsv', 'source': 'SymbolA', 'target': 'SymbolB', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': False}]
>>>
>>> # also possible without punctuation and spaces
>>> mapper.get_mappings(tags="BioPlex30293T", fields="name")
[{'name': 'BioPlex30293T', 'type': 'ppi', 'file_path': 'data/ppi/BioPlex_293T_Network_10K_Dec_2019.tsv', 'source': 'SymbolA', 'target': 'SymbolB', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': False}]
>>>
>>> # get both implemented BioPlex networks
>>> mapper.get_mappings(tags="BioPlex", fields="name")
[{'name': 'BioPlex30HCT116', 'type': 'ppi', 'file_path': 'data/ppi/BioPlex_HCT116_Network_5.5K_Dec_2019.tsv', 'source': 'SymbolA', 'target': 'SymbolB', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': False}, {'name': 'BioPlex30293T', 'type': 'ppi', 'file_path': 'data/ppi/BioPlex_293T_Network_10K_Dec_2019.tsv', 'source': 'SymbolA', 'target': 'SymbolB', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': False}]
>>>
>>> # get all implemented Gene regulatory networks
>>> mapper.get_mappings(tags="grn", fields="type")
[{'name': 'hetionetregulates', 'type': 'grn', 'file_path': 'data/hetionet/hetionet_regulates.tsv', 'source': 'GeneA', 'target': 'GeneB', 'sep': ' ', 'symbol': 'entrez', 'weight': 'None', 'directed': True}, {'name': 'GRNDBadrenalgland', 'type': 'grn', 'file_path': 'data/grndb/adrenal_gland.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBbloodx', 'type': 'grn', 'file_path': 'data/grndb/blood.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBbloodvessel', 'type': 'grn', 'file_path': 'data/grndb/blood_vessel.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBbrain', 'type': 'grn', 'file_path': 'data/grndb/brain.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBbreast', 'type': 'grn', 'file_path': 'data/grndb/breast.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBcolon', 'type': 'grn', 'file_path': 'data/grndb/colon.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBesophagus', 'type': 'grn', 'file_path': 'data/grndb/esophagus.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBheart', 'type': 'grn', 'file_path': 'data/grndb/heart.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBkidney', 'type': 'grn', 'file_path': 'data/grndb/kidney.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBliver', 'type': 'grn', 'file_path': 'data/grndb/liver.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBlung', 'type': 'grn', 'file_path': 'data/grndb/lung.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBmuscle', 'type': 'grn', 'file_path': 'data/grndb/muscle.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBnerve', 'type': 'grn', 'file_path': 'data/grndb/nerve.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBovary', 'type': 'grn', 'file_path': 'data/grndb/ovary.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBpancreas', 'type': 'grn', 'file_path': 'data/grndb/pancreas.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBpituitary', 'type': 'grn', 'file_path': 'data/grndb/pituitary.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBprostate', 'type': 'grn', 'file_path': 'data/grndb/prostate.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBsalivarygland', 'type': 'grn', 'file_path': 'data/grndb/salivary_gland.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBskin', 'type': 'grn', 'file_path': 'data/grndb/skin.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBsmallintestine', 'type': 'grn', 'file_path': 'data/grndb/small_intestine.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBspleen', 'type': 'grn', 'file_path': 'data/grndb/spleen.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBstomach', 'type': 'grn', 'file_path': 'data/grndb/stomach.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBtestis', 'type': 'grn', 'file_path': 'data/grndb/testis.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBthyroid', 'type': 'grn', 'file_path': 'data/grndb/thyroid.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDButerus', 'type': 'grn', 'file_path': 'data/grndb/uterus.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}, {'name': 'GRNDBvagina', 'type': 'grn', 'file_path': 'data/grndb/vagina.txt', 'source': 'TF', 'target': 'gene', 'sep': '\t', 'symbol': 'hgnc', 'weight': 'None', 'directed': True}]
>>>
>>> # get all implemented metabolic networks
>>> mapper.get_mappings(tags="metabolic", fields="type")
[{'name': 'Recon3D', 'type': 'metabolic', 'file_path': 'data/recon/reconparser/data/recon_directed.tsv', 'source': 'EntrezA', 'target': 'EntrezB', 'sep': '\t', 'symbol': 'entrez', 'weight': 'None', 'directed': True}]