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
    :inherited-members:

For example:

>>> import json
>>> from speos.preprocessing.mappers import GWASMapper
>>>
>>> # create an artificial mapping and save it.
>>> mapping = [ { "name": "foo",
...             "ground_truth": "bar.txt",
...             "phenotype": "bar",
...             "features_file": "foo.genes.out"} ]
>>> with open("mapping.json", "w") as file:
...     json.dump(mapping, file)
... 
>>> # init the mapper
>>> mapper = GWASMapper("ground_truth_dir", "feature_file_dir", "mapping.json")
>>>
>>> # retrieve all mapping with "foo" in their name
>>> mapper.get_mappings(tags="foo", fields="name")
[{'name': 'foo', 'ground_truth': 'ground_truth_dir/bar.txt', 'phenotype': 'bar', 'features_file': 'feature_file_dir/foo.genes.out'}]
>>>
>>> # retrieve all mapping
>>> mapper.get_mappings(tags="", fields="name")
[{'name': 'foo', 'ground_truth': 'ground_truth_dir/bar.txt', 'phenotype': 'bar', 'features_file': 'feature_file_dir/foo.genes.out'}]
>>>
>>> # returns empty list if requested mapping can not be found
>>> mapper.get_mappings(tags="xyz", fields="name")
[]