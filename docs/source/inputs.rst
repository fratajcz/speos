Usage
=====
Inputs
======

Speos automatically integrates various types of inputs, namely GWAS summary statistics, gene expression data and different biological networks.
To allow an extensible and easy integration of new data sources, please read the following documentation.


GWAS Data
---------

The mapping of phenotypes to appropriate GWAS traits is done by the ``speos.preprocessing.mappers.GWASMapper`` :

.. autoclass:: speos.preprocessing.mappers.GWASMapper
    :members:

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']
