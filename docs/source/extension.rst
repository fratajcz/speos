Extending Speos
===============

The strength of Speos lies in its ability to be customized and extended. See the following sections on how to add your own network, additional GWAS data and new phenotype labels!

Adding New Label Sets
---------------------

You might be aware that we have already implemented core genes for 20 different disorders. You can check  :doc:`here <./configuration>` if you are unsure what they are and how to use them.
However, chances are you have your very own set of assumed core genes that you want to use as known positives for training with Speos.

To do that, follow these instructions:

What Are Label Sets?
~~~~~~~~~~~~~~~~~~~~

Label sets are nothing other than a list of gene labels that are used as ground truth ("real") positives during training. In our minimal example here, our label set consists only of three genes which we store at ``extensions/labels.txt``:

.. code-block:: text

    BEND7
    POTEF
    AADAC

Note that these genes are represented by HGNC symbols. Other symbols, such as Entrez and Ensembl IDs are also possible. 

.. note:: 

    In this example we use only three positive genes for brevity. Be aware that our smallest tested label set contains roughly 120 genes. From how the training works we estimate that 100 genes should be enough, but more are definitely beneficial.

How Do I Add My Label Set?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you start, the file ``extensions/mapping.json`` should be empty except for an empty JSON list ("[]"). If the list is not empty but already contains a definition, just add to that list by seperating the entries (enclosed in "{}") with a comma. This is the place where we will put our definition for the label set.
Open the file and edit it to the following:

.. code-block:: json

    [{"name": "UNK-my_labels", 
      "ground_truth": "my_labels", 
      "features_file": "", 
      "function": "test_preprocess_labels", 
      "args": ["./extensions/labels.txt"], 
      "kwargs": {}
      }]

In this case, it is more or less irrelevant what you put in as :obj:`name` or :obj:`ground_truth`, as long as its not empty. The :obj:`features_file` is set to an empty string, unless you want to add GWAS data, which you can learn about further below.
Since Speos can't know how you structured the labels in your file, you have to give it a function that reads the label file and returns the labels as a python set. :obj:`args` and :obj:`kwargs` defines the arguments and keyword arguments which are fed to the function in order to return the right labels.
To make this work, all we now have to do is write the function :obj:`test_preprocess_labels` which reads :obj:`./extensions/labels.txt` from :obj:`args` and returns a set of HGNC identifiers. We add this function definition to the file :obj:`extensions/preprocessing.py`:

.. code-block:: python
    :linenos:
    :caption: extensions/preprocessing.py

    def test_preprocess_labels(path) -> set:
        import pandas as pd

        return set(pd.read_csv(path, sep="\t", header=None, names=["0"])["0"].tolist())

This function takes the path stored in :obj:`args`, reads the file, extracts the only column and transforms the contents into a set before returning them. You can test if it works as follows:

.. code-block:: console

    $ python
    Python 3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 06:08:21) 
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from extensions.preprocessing import test_preprocess_labels
    >>> test_preprocess_labels("./extensions/labels.txt")
    {'POTEF', 'BEND7', 'AADAC'}

Just what we wanted. Now, we can go ahead and actually use them for training.

How Do I Use My Label Set?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that the label sat "my_labels" has been added to the extensions, lets craft a config file to use them.
Lets create the file :obj:`config_my_labels.yaml` and fill it with the following content:

.. code-block:: text
    :linenos:
    :caption: config_my_labels.yaml

    name: test_my_labels

    input:
        tag: my_labels
        field: ground_truth

This will now sift through all extensions label definitions and inbuilt label definitions and return those with "my_labels" in the field :obj:`name`. Be careful to not use the same name twice, as duplicate entries are not allowed!

Lets test or newly added label set by running a quick training job:

.. code-block:: console

    $python training.py -c config_my_labels.yaml
    test_my_labels 2023-01-31 16:41:46,549 [INFO] speos.experiment: Starting run test_my_labels
    test_my_labels 2023-01-31 16:41:46,551 [INFO] speos.experiment: Cuda is available: False
    test_my_labels 2023-01-31 16:41:46,551 [INFO] speos.experiment: CUDA set to auto, no CUDA device detected, setting to CPU
    test_my_labels 2023-01-31 16:41:46,551 [INFO] speos.experiment: Using device(s): ['cpu']
    test_my_labels 2023-01-31 16:41:46,559 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30293T']
    test_my_labels 2023-01-31 16:41:46,560 [INFO] speos.preprocessing.preprocessor: Using 1 mappings with ground truth my_labels 
    Processing...
    test_my_labels 2023-01-31 16:41:57,257 [INFO] speos.preprocessing.preprocessor: Name: 
    Type: MultiDiGraph
    Number of nodes: 18638
    Number of edges: 185052
    Average in degree:   9.9287
    Average out degree:   9.9287
    Done!
    test_my_labels 2023-01-31 16:41:57,575 [INFO] speos.preprocessing.preprocessor: Number of positives in ground truth my_labels: 3
    test_my_labels 2023-01-31 16:41:58,066 [INFO] speos.preprocessing.datasets: Loading Processed Data from ./data/processed/test_my_labels.pt
    test_my_labels 2023-01-31 16:41:58,130 [INFO] speos.preprocessing.datasets: Data(x=[18638, 72], edge_index=[2, 185052], y=[18638], train_mask=[18638], test_mask=[18638], val_mask=[18638])
    test_my_labels 2023-01-31 16:41:58,214 [INFO] speos.experiment: Cuda is available: False
    test_my_labels 2023-01-31 16:41:58,214 [INFO] speos.experiment: CUDA set to auto, no CUDA device detected, setting to CPU
    test_my_labels 2023-01-31 16:41:58,289 [INFO] speos.experiment: Created new ResultsHandler pointing to ./results/test_my_labels.h5
    test_my_labels 2023-01-31 16:41:58,309 [INFO] speos.experiment: Received data with 3 train positives, 16771 train negatives, 0 val positives, 932 val negatives, 0 test positives and 932 test negatives

As you can see from the logging output: It worked! We now have three labeled positives. As you can see trom the last line, though, all our positives have been partitioned to the training set, leaving none for the validation and test sets.
This is of course impractical and would result in nonsense results. We therefore advise to have at least 100 true positives in your label set!

Additonal Networks
------------------

Although Speos already has a wide variety of networks that you can choose from, the field of Biology is so flexible and wide that it can be very handy to extend Speos so you can use additional networks.
You can add a network and use it in single or multi network training runs by simply adhering to a minimal header structure. The following example will guide you through the process.


Adding A Network
~~~~~~~~~~~~~~~~

Say you have an adjacency that you want to add to Speos to run some experiments on. For the sake of simplicity, lets say your edgelist contains only one edge and looks like this:

.. code-block:: text

    SymbolA SymbolB
    MTOR    IL1B

It describes a fictional connection between the MTOR and the IL1B gene using HGNC gene identifier. You now save this edgelist to :obj:`data/myadjacency/edgelist.tsv`.  
Now, the next step is to tell Speos how to use your new adjacency. To add an additional network to Speos, you can simply register it in :obj:`extensions/adjacencies.json`. Se the following example how to do it:

Without any manipulation, :obj:`extensions/adjacencies.json` simply contains an empty list:

.. code-block:: json
    :linenos:
    :caption: extensions/adjacencies.json

    []

This is because Speos has defined its core networks elsewhere (in :obj:`speos/adjacencies.json`).
To add a network, simply modify :obj:`extensions/adjacencies.json` as follows:


.. code-block:: json
    :linenos:
    :caption: extensions/adjacencies.json

    [{"name": "MyNetwork",
    "type": "ppi",
    "file_path": "data/myadjacency/edgelist.tsv",
    "source": "SymbolA",
    "target": "SymbolB",
    "sep": "\t",
    "symbol": "hgnc",
    "weight": "None",
    "directed": false}]

* :obj:`"name"`: This key specifies the name of the network, how it should be called in logging and plotting and with which name it should be matched during the search.
* :obj:`"type"` :This key specifiees the network's type and allows type-specific multi-network runs. Currently in use are "ppi", "grn", "evo" and "metabolic". If you set it to "ppi", you can blend it in with the other PPIs in a multi-network run. If you want to use you network in isolation, then this key is not important. If you want to add multiple networks, you can create your own type (i.e. "mytype") and use this to cluster your netowrks.
* :obj:`"file_path"`: Here you specify the path to the edgelist file starting in the speos main directory.
* :obj:`"source"` and :obj:`"target"`: Tese keys specify the column headers where the source and target nodes are specified for every edge.
* :obj:`"sep"`: This key specifies the column seperator of the file.
* :obj:`"symbol"`: This key specifies which type of symbol is used to identify the gene, use either "hgnc", "entrez" or "ensembl".
* :obj:`"weight"`: This key specifies if there is a column that contains edge weights. "None" means there are no edge weights (all have weight 1), otherwise specify the column header here. (not implemented yet)
* :obj:`"directed"`: This key contains a boolean (false/true) and tells Speos if the edges are directed or undirected.

Using your Network
~~~~~~~~~~~~~~~~~~

To use the network that we just have added to :obj:`extensions/adjacencies.json`, you can simply set according values in a config file and try to run a training run using that config file.

Select your network in the config :obj:`my_config.yaml`:

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

We can see that the network has been processed by looking at the logging output in the terminal. Our graph has 16852 nodes but only 2 edges! Why 2 edges and not just one? If you go up to our network definition, you will see that we set :obj:`"directed"` to :obj:`"false"`. 
This means that the edge can be traversed in both ways. Since we want to be able to both model directed and undirected edges without additional metadata, we have added 2 edges for our one undirected edge: One from MTOR to IL1B and one from IL1B to MTOR!

Using your Network together with others
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the :obj:`"type"` keyword in the description of the network to trigger a multi-network run. The :obj:`"type"` of our little netork is set to :obj:`"ppi"`, so we can blend it with other PPIs by using the folling config:

Adapt :obj:`my_config.yaml` to:

.. code-block:: text

    name: test_adjacency_multiple

    input:
        adjacency: ppi
        adjacency_field: type

and run it, which results in a different output:

.. code-block:: console

    $ python training.py -c my_config.yaml
    test_adjacency_multiple 2022-08-29 16:59:23,197 [INFO] speos.experiment: Starting run test_adjacency_multiple
    test_adjacency_multiple 2022-08-29 16:59:23,197 [INFO] speos.experiment: Cuda is available: True
    test_adjacency_multiple 2022-08-29 16:59:23,198 [INFO] speos.experiment: Using device(s): ['cuda:0']
    Processing...
    Done!
    test_adjacency_multiple 2022-08-29 16:59:23,202 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30HCT116', 'BioPlex30293T', 'HuRI', 'IntActPA', 'IntActDirect', 'MyNetwork']
    test_adjacency_multiple 2022-08-29 16:59:23,202 [INFO] speos.preprocessing.preprocessor: Using 8 mappings with ground truth ./data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    test_adjacency_multiple 2022-08-29 17:00:22,636 [INFO] speos.preprocessing.preprocessor: Name: 
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

Additonal GWAS Data
-------------------

By default, Speos integrates several GWAS traits and maps them to diseases according to the mapping found by `Freund et al. <https://www.sciencedirect.com/science/article/pii/S0002929718302854>`_. 
However, if multiple GWAS traits are mapped to the same disease, only the genes for which we have data for all of the GWAS traits can be used by Speos. For this reason, we have omitted some GWAS which had only sparse information across the genome.
It might be likely that in the future, GWAS with more participants uncover more loci which gives us information about more genes. Therefore, you might want to add more GWAS data to your analysis!

.. note::
   Before GWAS Data can be used in Speos, the SNP-level summary statistics has to be mapped to gene-level. This means that you need a P-Value, a Z-Value (Z-transformed P-Values) and the total number of SNPs per Gene to add you GWAS Data!

   We have used `MAGMA <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004219>`_ to aggregate the GWAS Data on gene-level, but there are other tools around.

Adding a GWAS Study
~~~~~~~~~~~~~~~~~~~~

Say you have a GWAS study that you want to add to Speos to run some experiments on. For the sake of simplicity, lets say your GWAS Data contains only information about 21 genes:

.. code-block:: text

    GENE CHR START STOP NSNPS NPARAM N ZSTAT P
    728378 1 586287 611297 7 3 20833 0.45359 0.32506
    100 1 803398 805130 4 2 20833 -1.6694 0.95249
    6137 1 925741 944581 57 18 20833 0.33094 0.37035
    222389 1 944203 959299 50 8 20833 -0.37583 0.64648
    5928 1 959952 965720 17 4 20833 -0.83635 0.79852
    25873 1 966497 975108 28 12 20833 0.37064 0.35545
    6124 1 975199 982117 22 6 20833 0.74433 0.22834
    6188 1 998962 1001285 12 4 20833 1.1117 0.13313
    708 1 1013467 1014540 6 2 20833 0.21337 0.41552
    375790 1 1020101 1056119 114 15 20833 -0.23388 0.59246
    105369174 1 1061207 1066390 10 5 20833 0.47219 0.31839
    105378948 1 1065635 1069326 10 3 20833 0.83152 0.20284
    401934 1 1071746 1074307 6 2 20833 0.95141 0.1707
    54991 1 1081818 1116356 115 15 20833 0.84384 0.19938
    254173 1 1173898 1197935 143 20 20833 3.2515 0.00057395
    8784 1 1203508 1206709 10 2 20833 3.2669 0.00054358
    7293 1 1211326 1214638 9 4 20833 4.0675 2.3761e-05
    51150 1 1216908 1232067 93 6 20833 1.8826 0.029875
    126792 1 1232249 1235041 7 2 20833 3.5157 0.00021927
    388581 1 1242446 1247218 13 3 20833 2.8675 0.0020689
    118424 1 1253912 1273854 42 7 20833 2.6434 0.0041033

This is the output of the `MAGMA <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004219>`_ tool from an actual GWAS trait which we have cropped and manipulated.
It has a GENE column which contains the Entrez ID of the gene, some optinal gene information and the COlumnds NSNPS, ZSTAT and P. These three columns are important and have to be present, alongside the GENE column.
They are seperated with a single blank space, not with a tab delimiter.

.. note::
   We allow much less flexibility in the GWAS Data file structure than with the adjacencies. 
   This is because we process them all identically with the same tool, so we just have to write one processing script. Edgelists and Networks come from various soruces, having various formats.

You now save this gene list to :obj:`data/mygwas/FOO.genes.out`.
Now, the next step is to tell Speos how to use your data. To add an additional GWAS trait to Speos, you can simply register it in :obj:`extensions/mapping.json`. Se the following example how to do it:

Without any manipulation, :obj:`extensions/mapping.json` simply contains an empty list:

.. code-block:: json
    :linenos:
    :caption: extensions/mapping.json

    []

This is because Speos has defined its GWAS data elsewhere (in :obj:`speos/mapping.json`).
To add a GWAS trait, simply modify :obj:`extensions/mapping.json` as follows:


.. code-block:: json
    :linenos:
    :caption: extensions/mapping.json

    [{"name": "FOO-immune_dysregulation",
    "ground_truth": "data/mendelian_gene_sets/Immune_Dysregulation_genes.bed",
    "phenotype": "immune_dysregulation",
    "features_file": "data/mygwas/FOO.genes.out",
    "match_type": "perfect",
    "significant": false}
    ]

* :obj:`"name"`: This key specifies the name of the mapping. It should contain the GWAS trait (FOO) and the disease it is mapped to (immune_dysregulation), seperated by a hyphen.
* :obj:`"ground_truth"` : This key specifies the name of the ground truth file that contains the labels of the disease that this GWAS trait is mapped to.
* :obj:`"phenotype"`: Here you specify the name of the phenotype/disease. This is only used for searching and logging.
* :obj:`"features_file"`: This key specifies the path to the GWAS data for your trait.
* :obj:`"match_type"`: This key specifies the type of match the trait has with the disease when it comes to symptoms. We adhere to the mapping from `Freund et al. <https://www.sciencedirect.com/science/article/pii/S0002929718302854>`_, where the symptoms of a trait can either match the disease with a "perfect" or a "related". If the trait would not match the symptoms of the disease at all, you would not include the mapping in the first place. This key can be used to filter traits.
* :obj:`"significant"`: This key specifies whether `Freund et al. <https://www.sciencedirect.com/science/article/pii/S0002929718302854>`_ have found a significant enrichment of genes that have a significant GWAS hit for this trait among the Mendelian disease genes. Since the trait FOO is made up, it is not included in their analysis and thus not significant.

Using your GWAS Study
~~~~~~~~~~~~~~~~~~~~~

Now that you have added your GWAS study to :obj:`extensions/mapping.json`, you can start using it. Note that we have specified the Immune Dysregulatin as ground truth and phenotype. If you look above in the :ref:`Using your Network` subsection, you will find the following line in the logging output:

.. code-block:: console

    ...
    test_adjacency 2022-08-29 16:43:17,432 [INFO] speos.preprocessing.preprocessor: Using 8 mappings with ground truth ./data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    ...

This means that by default, we have 8 GWAS traits that map to Immune Dysregulation. 

Now, lets write the following config file and save it to :obj:`my_config.yaml`:

.. code-block:: text
    :linenos:
    :caption: my_config.yaml

    name: test_gwas

    input:
        tag: Immune_Dysregulation
        field: ground_truth

This setting is also the default, but we define it anyway so that you know what to change if you want to run it for a differend ground truth. This settings means that it will look for the substring :obj:`Immune_Dysregulation` in the field :obj:`ground_truth` of all GWAS-to-disease-gene mappings and select all those that match.

Look what happens if we start a training run now after we have registered our FOO GWAS trait in :obj:`extensions/mapping.json`:

.. code-block:: console

    $ python training.py -c my_config.yaml
    test_gwas 2022-08-30 11:41:55,770 [INFO] speos.experiment: Starting run test_gwas
    test_gwas 2022-08-30 11:41:55,770 [INFO] speos.experiment: Cuda is available: True
    test_gwas 2022-08-30 11:41:55,770 [INFO] speos.experiment: Using device(s): ['cuda:0']
    Processing...
    Done!
    test_gwas 2022-08-30 11:41:55,773 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30293T']
    test_gwas 2022-08-30 11:41:55,773 [INFO] speos.preprocessing.preprocessor: Using 9 mappings with ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    test_gwas 2022-08-30 11:42:19,319 [INFO] speos.preprocessing.preprocessor: Name: 
    Type: MultiDiGraph
    Number of nodes: 18
    Number of edges: 18
    Average in degree:   1.0000
    Average out degree:   1.0000
    test_gwas 2022-08-30 11:42:19,344 [INFO] speos.preprocessing.preprocessor: Number of positives in ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed: 2

You see that this logging output is drastically different to the ones in the chapters above. First, it says :obj:`Using 9 mappings` instead of 8, so the additional trait FOO is being used. 
But then, our graph has only 18 nodes, even though we fed in GWAS data for 21 nodes for the trait FOO. This is because the remaining three nodes have either missing data for one of the other 8 traits, or there is no median gene expression data for these three.
In the last line, you can see that among these 18 nodes, only 2 positives (Mendelians) have been found. This is of course too few to construct a meaningful train, validation and test set, which is why the training run crashes soon after. 

This example should have shown you 1. how to add you own GWAS trait data and 2. that it is crucial that your GWAS trait has information about as many genes as possible.

.. note::
   Of course you can go ahead and simply impute p-value, Z-value and the number of SNPs for all the genes that have no information for your trait. In this case, just add the imputed values to :obj:`data/mygwas/FOO.genes.out` and re-run the analysis, now the number of used genes should be much larger.
   Since it is not clear how to impute such values, however, we will not advise to do so.

Additonal Node Features
-----------------------

In addition to GWAS trait data, Speos uses median gene expression per tissue as node level features. We are aware that there are plenty of other node feautures that can be used instead or in addition to those that are already implemented. The following example will lead you through the process of adding your own node data.

Adding your node feautures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Say you have some features that you can add to every node. For the sake of simplicity, let's assume we have three additional features for every gene, and each of those features are just the same three integers. This is pointless of course, but we don't want to get distracted by complicated examples.
The following is your data file that is stored in :obj:`"data/mydata/mydata.tsv"`:

.. code-block:: text
    :caption: data/mydata/mydata.tsv

    hgnc	Feat1	Feat2	Feat3
    A1BG	1	2	3
    A1CF	1	2	3
    A2M	1	2	3
    A2ML1	1	2	3
    A3GALT2	1	2	3
    A4GALT	1	2	3
    A4GNT	1	2	3
    AAAS	1	2	3
    AACS	1	2	3
    AADAC	1	2	3
    AADACL2	1	2	3
    AADACL3	1	2	3
    ...

And so on, the same three features for every gene, preceded by the HGNC gene symbol.

next, we need to write a function that processes this file and returns it as a pandas DataFrame. We are aware that the preprocessing in this case is trivial, but since there can be arbitrary types of input, we want to give the user the chance to use any input by not making any assumptions.
You can write any preprocessing function that you want, as long as it returns a pandas DataFrame where the row index is either the HGNC, Entrez or Ensembl identifiers.

For the file shown above, we write this simple preprocessing script and place it in :obj:`"extensions/preprocessing.py"`:

.. code-block:: python
    :linenos:
    :caption: extensions/preprocessing.py

    def preprocess_mydata(path):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=0, index_col=0)

    return df

which, when run with the path to the file, returns the dataframe in the proper format:

.. code-block:: console

    $ python
    Python 3.7.12 | packaged by conda-forge | (default, Oct 26 2021, 06:08:21) 
    [GCC 9.4.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from extensions.preprocessing import preprocess_mydata
    >>> preprocess_mydata("data/mydata/mydata.tsv")
            Feat1  Feat2  Feat3
    hgnc                        
    A1BG         1      2      3
    A1CF         1      2      3
    A2M          1      2      3
    A2ML1        1      2      3
    A3GALT2      1      2      3
    ...        ...    ...    ...
    ZYG11A       1      2      3
    ZYG11B       1      2      3
    ZYX          1      2      3
    ZZEF1        1      2      3
    ZZZ3         1      2      3

    [19220 rows x 3 columns]
    >>> exit()

Now, all that is left to do is tell Speos to use the data. To do that, we add some descriptive keys to :obj:`"extensions/datasets.json.py"`.

Before manipulation, :obj:`"extensions/datasets.json.py"` looks like this:

.. code-block:: json

    []

Now, to add or dataset, we have to make the following additions:


.. code-block:: json

    [{"name": "MyData",
      "identifier": "hgnc",
      "function": "preprocess_mydata",
      "args": [],
      "kwargs": {"path": "data/mydata/mydata.tsv"}
      }]

* :obj:`"name"`: This key specifies the name of the dataset. It is only used for logging, so use something descriptive.
* :obj:`"identifier"` : The identifier that is used in the dataset file. It is allowed to use "hgnc", "entrez" or "ensembl".
* :obj:`"function"`: The name of the function that has been placed in :obj:`"extensions/preprocessing.py"` and that should be used to preprocess the data.
* :obj:`"args"` and :obj:`"kwargs"`: These keys are the arguments and keyword arguments for the preprocessing function chosen in the :obj:`"function"` key. Here, we need to pass only the path, but you can use any degree of customization in your preprocessing.


Using Your Node Feautures
~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have successfully registered the additional Dataset, it is used automatically. To demonstrate, let's start a simple training run.
Write a config and store it under :obj:`my_config.yaml`, containing the following lines:

.. code-block:: text

    name: test_input

    input:
        adjacency: ppi
        adjacency_field: type

And now we run it:

.. code-block:: console

    $ python training.py -c my_config.yaml
    test_input 2022-08-30 14:28:32,149 [INFO] speos.experiment: Starting run test_input
    test_input 2022-08-30 14:28:32,149 [INFO] speos.experiment: Cuda is available: True
    test_input 2022-08-30 14:28:32,149 [INFO] speos.experiment: Using device(s): ['cuda:0']
    test_input 2022-08-30 14:28:32,152 [INFO] speos.preprocessing.preprocessor: Using Adjacency matrices: ['BioPlex30293T']
    test_input 2022-08-30 14:28:32,152 [INFO] speos.preprocessing.preprocessor: Using 8 mappings with ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed 
    Processing...
    Done!
    test_input 2022-08-30 14:28:32,152 [INFO] speos.preprocessing.preprocessor: Using 1 additional node data sources: ['MyData']
    test_input 2022-08-30 14:28:58,192 [INFO] speos.preprocessing.preprocessor: Name: 
    Type: MultiDiGraph
    Number of nodes: 16852
    Number of edges: 158962
    Average in degree:   9.4328
    Average out degree:   9.4328
    test_input 2022-08-30 14:28:58,593 [INFO] speos.preprocessing.preprocessor: Number of positives in ground truth data/mendelian_gene_sets/Immune_Dysregulation_genes.bed: 523
    ...
    test_input 2022-08-30 14:28:59,330 [INFO] speos.datasets: Data(x=[16852, 99], edge_index=[2, 158962], y=[16852], train_mask=[16852], test_mask=[16852], val_mask=[16852])
    ...

You can see the line :obj:`Using 1 additional node data sources: ['MyData']` indicating that it finds the definition of your dataset. 
Further down you can see the dimension of the feature matrix: :obj:`Data(x=[16852, 99], ...)` indicating that we have 16852 genes which each has 99 features. 

If we delete our description from :obj:`extensions/datasets.json.py` (i.e. turn it into an empty list again), and leave everything else as it is, the corresponding line in the output will change to:

.. code-block:: console

    test_input 2022-08-30 14:58:46,339 [INFO] speos.datasets: Data(x=[16852, 96], edge_index=[2, 158962], y=[16852], train_mask=[16852], test_mask=[16852], val_mask=[16852])

And the part :obj:`Data(x=[16852, 96], ...)` indicates that, without our "MyDataset", we have only 96 features. So, adding the 3 features beforehand was a success!

Extending Postprocessing
------------------------

As you might have seen in our manuscript, there are several steps happening after the training of the models. The first is the establishment of model concordance, or overlap. This postprocessing step yield the convergence scores
and is therefore independent of which disease we are looking at, only dependent on the predictions of the models.

The rest of the postprocessing steps are external validations. Some of those, like Loss of Function (LoF) intolerance enrichment or drug target enrichment, are also disease-agnostic. A gene has a specific LoF intolreance Z-score that does not change wheter we look for genes for cardiovascular disease, immune dysregulation or any other disease.
Thus, these external validations will not need any extensions to run, even if you completely customized the rest of Speos.

However, some external validations, such as the enrichment of differentially expressed and mouse KO genes, requires disease dependent gene sets. In the following we will show you how you can add these sets for your own customized Speos runs.

Adding Mouse Knockout Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

We obtained mouse knockout genes from the `Mouse Genome Database <http://www.informatics.jax.org/allele>`_, for more method details consult the method section of our `preprint <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1.full.pdf>`_. 

Mouse Knockout data is matched to the disease of a given run by the file :obj:`data/mgi/query_mapping.yaml`. An excerpt of this file shows you how the mapping has to look like:

.. code-block:: text
    :linenos:
    :caption: data/mgi/query_mapping.yaml (excerpt)

    "cardiovascular_disease":
        file: "./data/mgi/cad_query.txt"
    "immune_dysregulation":
        file: "./data/mgi/immune_dysreg_query.txt"

In every line, a disease tag :obj:`"cardiovascular_disease"` is mapped to a file :obj:`"./data/mgi/cad_query.txt"` in yaml format. So, if you added your disease with the tag :obj:`my_disease` and want to add the mouse knockout genes obtained from the `MGI Database <https://www.informatics.jax.org/allele>`_ and saved at :obj:`data/mgi/my_disease_query.txt` then add the following lines to :obj:`data/mgi/query_mapping.yaml`:

.. code-block:: text
    :linenos:
    :caption: data/mgi/query_mapping.yaml (continued)

    "my_disease":
        file: "./data/mgi/my_disease_query.txt"

and with the next run your freshly added mouse KO genes will automatically be selected for :obj:`my_disease`.

Adding Differential Gene Expression Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We obtained differntially expressed genes from `GEMMA database <https://gemma.msl.ubc.ca/phenotypes.html>`_, for more method details consult the method section of our `preprint <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1.full.pdf>`_. 

Differential gene expression data is matched to the disease of a given run by the file :obj:`data/dge/mapping.yaml`. An excerpt of this file shows you how the mapping has to look like:

.. code-block:: text
    :linenos:
    :caption: data/dge/mapping.yaml (excerpt)

    "cardiovascular_disease":
        "Coronary Artery Disease":
            file: "./data/dge/cad.gemma"
        "Atrial Fibrillation":
            file: "./data/dge/af.gemma"
        "Aortic Aneurysm":
            file: "./data/dge/aa.gemma"
        "Ischemia":
            file: "./data/dge/is.gemma"
        "Hypertension":
            file: "./data/dge/hy.gemma"
        "Atherosclerosis":
            file: "./data/dge/ar.gemma"
    "immune_dysregulation":
        "Crohn's Disease":
            file: "./data/dge/cro.gemma"
        "Ulcerative Colitis":
            file: "./data/dge/ulc.gemma"
        "Lupus Erythematosus":
            file: "./data/dge/lup.gemma"
        "Rheumatoid Arthritis":
            file: "./data/dge/rhe.gemma"
        "Multiple Sclerosis":
            file: "./data/dge/ms.gemma"

In every line, a disease tag :obj:`cardiovascular_disease` is mapped to an array of disease subtypes, each linking to a file. 
So, if you added your disease with the tag :obj:`my_disease` and want to add differentially expressed genes obtained for the subtypes :obj:`Subtype A` and :obj:`Subtype B` from `GEMMA database <https://gemma.msl.ubc.ca/phenotypes.html>`_ and saved at :obj:`./data/dge/suba.gemma` and :obj:`./data/dge/subb.gemma` then add the following lines to :obj:`data/dge/mapping.yaml`:

.. code-block:: text
    :linenos:
    :caption: data/dge/mapping.yaml (continued)

    "my_disease":
        "Subtype A":
            file: "./data/dge/suba.gemma"
        "Subtype B":
            file: "./data/dge/subb.gemma"

and the next time you run preprocessing, the results will automatically contain your new enrichment analysis for differentially expressed genes!

Adding New Tasks
~~~~~~~~~~~~~~~~

For now, configuring the postprocessor so that users can implement and run their own tasks is on the TODO list. If you want the option to add your own task, let us know by filing a feature request via `GitHub Issues <https://github.com/fratajcz/speos/issues>`_ so we know that this features is needed and can bump up its priority.