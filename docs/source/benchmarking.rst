Benchmarking
============

Often in machine learning applications, a considerable amount of effort is placed on finding the right models and hyperparameters. While it is generally possible to make use of general hyperparameter search frameworks liek `scikit-learn's ParamterGrid <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html>`_ in order to manipulate Speos' configs and thus create a blueprint for a hyperparmeter search, we also have inbuilt benchmarking capabilities directly in Speos.

To use the unbuilt benchmarking feautures, you will need to wrte two files, a config file that contains the shared settings between all runs (i.e. the label set etc.) and a parameter file which details the individual runs and which settings should deviate from the shared settings in which run. Let's come up with a simple benchmarking case together.

Configuring Settings of Runs
----------------------------

Let's say you want to predict core genes for the ground truth gene set of Cardiovascular Disease, and you want to use BioPlex 3.0 293T as network. What you want to find out is which graph convolution works best with these two fixed settings. Let's first draft the config that makes sure we use the right shared settings:

.. code-block:: text
    :caption: config_cardiovascular_bioplex.yaml
    :linenos:

    name: cardiovascular_bioplex

    crossval:
        n_folds: 4

    input:
        adjacency: BioPlex30293T
        tag: Cardiovascular_Disease

    model:
        pre_mp:
            n_layers: 2
        mp:
            n_layers: 2
        post_mp:
            n_layers: 2     

Save these settings in ```config_cardiovascular_bioplex.yaml```. We have now defined our input and the model depth: The preprocessing network, the message passing network and the postproscessing network are all defined as having a depth of two, each.

Now, let's define our parameter file which contains the settings that should change between the individual runs:

.. code-block:: text
    :linenos:
    :caption: parameters_layers.yaml

    name: layers

    metrics: 
    - mean_rank_filtered
    - auroc
    - auprc

    parameters:
    -   name: gcn
        model:
            mp:
                type: gcn
    -   name: gat
        model:
            mp:
                type: gat
    -   name: gin
        model:
            mp:
                type: gin
    -   name: graphsage
        model:
            mp:
                type: sage
    -   name: mlp
        model:
            mp:
                n_layers: 0

and save these settings as ```parameters_layers.yaml```. The first `name` tag defines the name of the while benchmarking array and should be descriptive of what this array is about. then, the `metrics` section defines an array of metrics that should be obtained and recorded for these runs.
The `parameters` section is where it gets interesting. It contains a list of mini-configs, each with an individual `name` tag that describes this individual parameter setting, followed by the settings which should be changed from the shared settings for this indivudal benchmark run.
As you see, we have four different graph convolutions selected and now want to see which of those layers provides the best performance, as measured by the three metrics we have chosen. The last parameter setting, `mlp`, answers the question about the performance difference if we use no graph convolution at all, therefore we have set the `n_layers` tag for the message passing module to 0, leaving only the fully connected layers in pre- and post-message passing.
While this might not directly answer our question which convolution is best, it is always important to have a contrast setting in case *no* convolution is actually the best.

Starting a Benchmark Run
------------------------

You can now go ahead and start a benchmark run from the command line:

.. code-block:: console

    python benchmarking.py -c config_cardiovascular_bioplex.yaml -p parameters_layers.yaml

This will start a 4-fold crossvalidation for each of the total of five parameter settings that we have described above. For statistical rigor, each fold is repeated 4 times, so that we obtain 4 * 4 * 5 = 80 models in total, 16 per parameter setting.

Each of the runs has an individual name, such as ```cardiovascular_bioplex_layers_gcn_rep0_fold0```, which is put together from the individual name tags of config, parameter file, parameter setting, repetition and fold. You can watch the output of the benchmark run to see the changes your settings make.

For example, for the first 16 models, the model description in the logging output should look like the following:

.. code-block:: text
    :caption: logging output

    [...]

    cardiovascular_bioplex_layers_gcnrep0_fold_0 2023-02-10 14:18:29,616 [INFO] speos.experiment (0): GeneNetwork(
    (pre_mp): Sequential(
        (0): Linear(96, 50, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(50, 50, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(50, 50, bias=True)
        (5): ELU(alpha=1.0)
    )
    (post_mp): Sequential(
        (0): Linear(50, 50, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(50, 50, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(50, 25, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(25, 1, bias=True)
    )
    (mp): Sequential(
        (0): GCNConv(50, 50)
        (1): ELU(alpha=1.0)
        (2): InstanceNorm(50)
        (3): GCNConv(50, 50)
        (4): ELU(alpha=1.0)
        (5): InstanceNorm(50)
    )
    
    [...]

While for subsequent runs, the ```(mp)``` part should change, for example to:

.. code-block:: text
    :caption: logging output (continued)

    [...]

    cardiovascular_bioplex_layers_gatrep0_fold_0 2023-02-10 14:42:13,746 [INFO] speos.experiment (0): GeneNetwork(

    [...]

    (mp): Sequential(
        (0): GATConv(50, 50, heads=1)
        (1): ELU(alpha=1.0)
        (2): InstanceNorm(50)
        (3): GATConv(50, 50, heads=1)
        (4): ELU(alpha=1.0)
        (5): InstanceNorm(50)
    )

    [...]

Which shows that in the second setting, the GCN layers have been replaced by GAT layers!

Evaluating the Benchmark
------------------------

Once your benchmark is finished, you should end up with a results file that contains detailed performance results for all models and metrics. In our case, it is called ```cardiovascular_bioplex_layers.tsv```` and should look more or less like this:

.. code-block:: text
    :linenos:
    :caption: cardiovascular_bioplex_layers.tsv

    	mean_rank_filtered	auroc	auprc
    cardiovascular_bioplex_layers_gcnrep0_fold0	6058.653846153846	0.6573416453300679	0.016763970872865765
    cardiovascular_bioplex_layers_gcnrep0_fold_1	3677.68	0.7798784370477568	0.04156103030017034
    cardiovascular_bioplex_layers_gcnrep0_fold_2	6085.64	0.6379849449913145	0.0181314460851689
    cardiovascular_bioplex_layers_gcnrep0_fold_3	4449.44	0.7437521713954836	0.019851396078300727
    cardiovascular_bioplex_layers_gcnrep1_fold_0	5781.961538461538	0.6683179338750974	0.017543154801648458
    cardiovascular_bioplex_layers_gcnrep1_fold_1	3671.04	0.7760115774240232	0.029372573541881476

you can now go ahead, read the table and produce some informative figures. Since you know that we have 16 models per setting, each 16-row block belongs to one setting. Here is the necessary code in python:

.. code-block:: python
    :linenos:

    import pandas as pd 
    import matplotlib.pyplot as plt

    results = pd.read_csv("cardiovascular_bioplex_layers.tsv", sep="\t", header=0)
    methods = ["GCN", "GAT", "GIN", "GraphSAGE", "MLP"]
    mean_ranks = []
    auroc = []
    auprc = []

    stride = 16

    for start in range(0, len(results), stride):
        method_results = results.iloc[start:start+stride, :]
        mean_ranks.append(results["mean_rank_filtered"])
        auroc.append(results["auroc"])
        auroc.append(results["auprc"])

    fig, axes = plt.subplots(3, 1)

    metrics = [mean_ranks, auroc, auprc]
    metric_names = ["Mean Rank (filtered)", "AUROC", "AUPRC"]

    for ax, metric, name in zip(axes, metrics, metric_names):
        ax.grid(True, zorder=-1)

        for i, run in enumerate(metric):
            jitter = np.random.uniform(-0.2, 0.2, len(run)) + i
            bp = ax.boxplot(run, positions=[i], widths=0.8, showfliers=False, zorder=1)
            ax.scatter(jitter, run, zorder=2)

        ax.set_ylabel(name)
        ax.set_xticks(range(len(methods)), methods)
        ax.set_xlabel('Method')
    
    plt.tight_layout()
    plt.savefig("benchmark_cardiovascular_bioplex_layers.png", dpi=350)

Which produces the following figure:


For mean rank, lowest is best, while for AUROC and AUPRC, highest is best. As you can see, the MLP clearly performs best, with GraphSAGE being a close follower. This is likely due to GraphSAGEs ability to seperate the self-information from the neighborhood information and thus being aple to replicate an MLP.
As we can see here relatively clearly, the network that we have chosen, Bioplex 3.0 293T, is not very favorable for the selected graph convolutions, as the MLP which does not use it performs best. 

With this type of analysis, it is fast and easy to ascertain which parts of the input or neural network should be placed more attention upon. Here, using a different network or tesiting a wider range of graph convolutions might improve performance.


