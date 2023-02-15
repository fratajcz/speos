High-Level API
==============

Speos has a high-level API that lets you run jobs, do postprocessing and even hyperparameter search with only a config file and a call to the respective pipeline.
For convenience, we have packaged the pipelines in command-line scripts so that you don't have to get your fingers dirty at all.

Let's first come up with an example config file. The whole list of settings that you can manage and their default values in your config file is shown in the `config_default.yaml <https://github.com/fratajcz/speos/blob/master/speos/utils/config_default.yaml>`_

Lets write a config file like the following:

.. code-block:: text

    name: myrun

    input:
      adjacency: BioPlex30293T
      tag: Cardiovascular Disease

    model:
      mp:
        type: gcn

and save this rudimentary config as :obj:`my_config.yaml`. 

Training
--------

If we just want to trigger a single training run, say, to check if our config does what we want, we can pass it to the high-level training script.


.. code-block:: console

  $ python training.py -c my_config.yaml

This will trigger a training run using the Mendelian genes for Cardiovascular Disease as ground truth labels, BioPlex 3.0 293T as adjacency and a vanilla GCN as graph convolution.
The rest of the settings will be default. 

After the model has been trained and early stopped on the holdout set, an Inference will be triggered and predictions for all genes will be produced. The settings for the inference, like cutoff value or save path,
are defined in the default config. 

Inference Only
--------------

Let's say you just ran the training command above but you can't find the directory where the results are saved, or the results have been deleted.
We can modify the config from above to explicitely tell Speos to save inference results to a specific directory by adding the following lines:

.. code-block:: text

    inference:
        save_dir: ~/results/
        save_sorted: True
        save_tsv: True

and re-run only the inference:

>>> python inference.py -c my_config.yaml

which will save all the results of the inference to :obj:`~/results/` without training the model again.
This assumes that the model has not been moved/deleted since it has been trained and that the config settings for the model save path are the same like when it was trained.

The Nested Crossvalidation
--------------------------

Although training a single model is nice and helpful, especially for testing purposes, Speos stands out by its nested crossvalidation scheme.
It allows you to train an ensemble of methods and use the overlap in their predictions to prioritize disease genes.
Afterwards, it uses these prioritized gene sets and the ground truth and does postprocessing analyses such as differentially expressed genes or drug target enrichment.

Let's first draft a suitable config file:

.. code-block:: text

    name: mycrossval

    input:
      adjacency: BioPlex30293T
      tag: Cardiovascular Disease

    model:
      mp:
        type: gcn

    crossval:
      mode: kfold
      n_folds: 10
      positive_only: True
    
    inference:
      save_dir: ~/results/
      save_sorted: True
      save_tsv: True

and save it as :obj:`my_crossval_config.yaml`. 

Then we only have to start the crossvalidation run: 

>>> python outer_crossval.py -c my_crossval_config.yaml

and wait for the results to roll in. Keep in mind that this trains n * (n + 1) = 110 models, so it might take a while on a cpu-only machine. 
Luckily, Speos auto-detects available cuda devices by default and moves the training and inference over to the gpu automatically.

Post-Processing In Detail
-------------------------

So, the ensemble mentioned above has been trained but you can't find the results of the postprocessing?
Let's have a closer look at the default values for our postprocessing routine (as defined in the `config_default.yaml <https://github.com/fratajcz/speos/blob/master/speos/utils/config_default.yaml>`_):

.. code-block:: text

    pp:                           # postprocessing
    switch: on                  # on, off, True, False
    tasks: [overlap_analysis, dge, pathway, hpo_enrichment, go_enrichment, drugtarget, druggable, mouseKO, lof_intolerance]   # this is the full set of postprocessing options
    consensus: top_down                # either int specifying the min bin for consensus or bottom_up, or top_down for p-val search starting from 0 up or from 10 down
    cutoff_value: 0.7           # float in case of cutoff_type split, else int
    cutoff_type: split          # split, top or bottom
    save: True
    save_dir: ./results/
    plot: True
    plot_dir: ./plots/

Let us walk through the keys one by one.

:obj:`switch: on`: 
    This is easy, setting this value to off or False disables postprocessing
:obj:`tasks: [...]`: 
    This describes the tasks that should be done on the ensemble.
    Here, overlap_analysis is necessary to find the convergence properties of the ensembles and count the votes for every gene.
    Without this task, the postprocessing won't work.
    The other tasks are pretty self-explanatory. If, for example, you would like to only do the drug target analysis and not waste time on the other tasks, reformat add the following lines to your :obj:`my_crossval_config.yaml`:

    .. code-block:: text

        pp:
          tasks: [overlap_analysis, drugtarget]

    and it will skip the other tasks.

:obj:`consensus: top_down/bottom_up`:
    This key regulates the application of the consensus score. :obj:`top_down` means that it will start at bin 10 (unanimous decision) and go down until it finds the first bin that is not significantly enriched for the positive holdout set. 
    The consensus score will be the last bin that is still significant. :obj:`bottom_up` means that it will start at bin 1 and go up until it reaches the first bin that is significant. This bin will be the consensus score.

:obj:`cutoff_value: (float)/(int)`: 
    A number between 0 and 1, indicating the cut-off of uncalibrated probabilities assigned to the genes. 
    In other words, setting it to 0.7 means that all genes with a prediction higher than 0.7 will receive a vote from this model. 
    Increasing the cutoff value decreases the number of genes that are voted to be disease genes.
    Can also be an integer above 1, in case :obj:`cutoff_type`: is set to :obj:`top/bottom`.

:obj:`cutoff_type: split/top/bottom`: 
    indicating the way that the cutoff is applied. :obj:`split` means that the predictions are split at the given float, as explained right above. 
    :obj:`top` means that the top :obj:`k` genes are selected, but the the :obj:`cutoff_value` has to be an integer (i.e. 100 as in Schulte-Sasse et al.)
    :obj:`bottom`means that the bottom :obj:`k` genes are ignored and the rest is selected.

:obj:`save: True`
:obj:`save_dir: ./results/`
:obj:`plot: True`
:obj:`plot_dir: ./plots/`
These keys are pretty self-explanatory. So, if you have lost your results or ask yourself where the plots went, this is where you should look.



Now, if you think that you want to try some changes in these settings, i.e. a different :obj:`cutoff_value` or different :obj:`tasks`, just add the respective lines to your :obj:`my_crossval_config.yaml` and re-run the postprocessing only:

>>> python postprocessing.py -c my_crossval_config.yaml

Have a little fun with it, but don't forget that hyperparameter hacking and repeated hypothesis testing with different settings like this weakens your evidence!