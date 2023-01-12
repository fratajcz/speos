Configuring Speos
=================

As you might have noticed, most of Speos' funcitonality can be configured using a few settings in a config file and the respective pipeline. 

So, lets first go through the most basic settings that let you configure the essentials.

.. note::
   In the following chapters, we will repeatedly refer to the default config in `speos/utils/config_default.yaml`. This file holds the complete set of settings and MUST NEVER BE ALTERED! 

   Instead, each time you want to start a run with new settings, create a new config file including the keys you want to override from the default settings.

Input Data
----------

There are several ways in which the input data can be changed using the respective keys in the config file. Lets go through the most important settings you should know about.

Adjacency Matrices (Networks)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the default config (excerpt):

.. code-block:: text

    input:
        adjacency: BioPlex30293T
        adjacency_field: name
        adjacency_blacklist: [recon3d, string]
 
These settings control the adjacency matrices used for construction of the GNN. All settings here are relative to the adjacency definitions provided in `speos/adjacencies.json` and, if you added some of your own, in `extensions/adjacencies.json`.
By using the settings above, each entry that contains the value "BioPlex30293T" in the field "name" in those two files will be used for the GNN message passing. Standard python string matching in lower case is possible, so just putting "bioplex" will give you both BioPlex networks.
You can, however, use any field in the two files for matching. If you want to use all Protein-Protein-Interaction (PPI) networks, set `adjacency: ppi` and `adjacency_field: type`. Using `adjacency: all` will give you all networks that arent explicitely blacklisted.
`adjacency_blacklist` lets you control what should not be used, even if it matches the criteria set above. As you see, recon3d and string are not used, even though they are included in the framework. String was excluded due to the degree distribution, and recon was simply added after all experiments were performed. You can already use it, but since we want Speos to recreate our experiments by default, it is on the blacklist for the time being.

Node Features
~~~~~~~~~~~~~

.. code-block:: text

        gwas_mappings: ./speos/mapping.json
        tag: Immune_Dysregulation
        field: ground_truth
        use_gwas: True                        # if gwas features should be used
        use_expression: True                  # if tissue wise gene expression values should be used
        use_embeddings: False                 # if the embeddings obtained from node2vec should be concatenated to the input vectors (laoded from embedding_path)
        embedding_path: ./data/misc/walking_all.output

`gwas_mappings` controls which GWAS trait is mapped to which set of Mendelian disorder genes. `tag` and `field` control which gwas-to-disease mappings should be used for the run. In this case, all GWAS traits that are mapped to the Mendelian disorder genes matching "Immune_Dysregulation" are used as input.
Changing this field to "Cardiovascular" will make Speos use different positive labels and different, matching GWAS traits!
`use_gwas` and `use_expression` are boolean flags that control if the respective type of input features are used for this run. This is useful for ablation studies.
`use_embeddings` lets you include the pre-trained node embedding vectors using Node2Vec. You can also train them yourself and add them with the key `embedding_path`.

This should give you a good overview on how to customize the input data of your runs.