Training the Ensemble
=====================

As mentioned earlier, The heart of Speos is the model ensemble consisting of a nested crossvalidation and the postprocessing that takes the predictions of all models, assesses the overlap and returns a set of candidate genes.
For details on how the data is partitioned between the individual models and how the overlap is compared to a statistical cutoff, please consult the methods section in our `preprint <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1.full.pdf>`_.

Running an Outer Crossvalidation Ensemble
-----------------------------------------

After testing several hyperparameters, networks and input variants via :doc:`Benchmarking <benchmarking>` and selecting a promising combination, it is time to run the full outer crossvalidation. Keep in mind that this will train n*(n+1) = 110 models with default settings.

First, lets draft a suitable config file:

.. code-block:: text
  :linenos:
  :caption: cardiovascular_gcn.yaml

    name: cardiovascular_gcn

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

and save it as :obj:`cardiovascular_gcn.yaml`. 

Here we use mostly the same settings as in the :doc:`API <api>` section earlier, you should adapt it to the settings that produced the best results in your benchmark runs. The only keys that should be identical to the config above is everything below :obj:`crossval` and :obj:`inference`.

Then, you run it with the following command:

.. code-block:: console

  $python outer_crossval.py -c my_crossval_config.yaml

And wait for the results to roll in. At the end of the ensemble training, the postprocessor will gather the results and asses the predictions in overlaps. You can find this section towards the end of the logfile:

.. code-block:: text
    :linenos:
    :caption: logs/mycrossval
  
    cardiovascular_gcn 2023-02-22 14:43:50,309 [INFO] speos.postprocessing.postprocessor: Applying concensus strategy: top_down
    cardiovascular_gcn 2023-02-22 14:43:50,309 [INFO] speos.postprocessing.postprocessor: Starting Overlap Analysis.
    cardiovascular_gcn 2023-02-22 14:44:19,159 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_0_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:44:44,929 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_1_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:45:06,524 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_2_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:45:44,366 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_3_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:46:03,944 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_4_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:46:29,244 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_5_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:46:45,441 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_6_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:47:07,211 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_7_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:47:36,267 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_8_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:48:07,654 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_9_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:48:34,821 [INFO] speos.postprocessing.postprocessor: Plotting overlap plot to cardiovascular_gcn_outer_10_fold__overlap.svg
    cardiovascular_gcn 2023-02-22 14:48:35,133 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #0: 7; Returned 560 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,147 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #1: 8; Returned 351 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,167 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #2: 7; Returned 352 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,184 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #3: 8; Returned 428 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,199 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #4: 6; Returned 559 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,222 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #5: 8; Returned 284 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,238 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #6: 6; Returned 558 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,255 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #7: 7; Returned 425 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,273 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #8: 7; Returned 287 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,332 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #9: 8; Returned 252 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,356 [INFO] speos.postprocessing.postprocessor: Consensus Score for Outer Crossval #10: 6; Returned 512 Candidate Genes
    cardiovascular_gcn 2023-02-22 14:48:35,431 [INFO] speos.postprocessing.postprocessor: Outer Crossvalidation results in 781 candidate genes in total. Results written to ./results/cardiovascular_gcnouter_results.json

As we see, each of the n+1=11 outer crossvalidation folds produced an overlap plot, from which a consensus score (CS) has been chosen to arrive at an overlap cutoff which in the end results in a set of candidate genes for each outer fold.  In the last line, we see that the union of the 11 sets contains 187 candidate genes, which means that several genes must have been predicted by more than one outer crossvalaidation fold.
How often each of the candidate genes has been predicted, i.e. its CS, can be seen in the JSON file that the log refers to:


.. code-block:: text
    :linenos:
    :caption: ./results/cardiovascular_gcnouter_results.json (beginning)

    [
        {
            "A1BG": 9,
            "A2M": 9,
            "ACAA2": 11,
            "ACO2": 11,
            "ACSL1": 9,
            "ACTB": 6,
            "ACTG2": 5,
            "ADH1B": 10,
            "ADH4": 11,
            "ADH5": 1,
            "ADI1": 5,
            [...]

We see the sorted HGNC symbols of the 781 candidate genes accompanied by their CS. The higher the CS, the more outer crossvalidations have predicted the gene to be a candidate, i.e. the higher the confidence. 

At the very end of the file, you will find summary statistics of how many genes have received which CS:

.. code-block:: text
    :linenos:
    :caption: ./results/cardiovascular_gcnouter_results.json (end)

        {
            "9": 48,
            "11": 163,
            "6": 40,
            "5": 41,
            "10": 66,
            "1": 161,
            "3": 58,
            "7": 39,
            "2": 89,
            "8": 37,
            "4": 39
        }
    ]

As we see, a total of 163 genes has received a CS of 11, which is more than the intermediate CS of 6, 7 and 8. 

The next lines in the config file belong to the external validation tasks which will be explained in the next chapter. First, lets take a look at the settings in the config with which you can change the behaviour of the postprocessor and how it produces candidate genes:

Configuring the Postprocessor
-----------------------------

There are plenty of ways to adapt the postprocessing. We encourage you to stick to the defaults first, but you might want to tweak your settings later.

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

* :obj:`switch: on`: 
    This is easy, setting this value to off or False disables postprocessing
* :obj:`tasks: [...]`: 
    This describes the tasks that should be done on the ensemble.
    Here, overlap_analysis is necessary to find the convergence properties of the ensembles and count the votes for every gene.
    Without this task, the postprocessing won't work.
    The other tasks are pretty self-explanatory. If, for example, you would like to only do the drug target analysis and not waste time on the other tasks, reformat add the following lines to your :obj:`my_crossval_config.yaml`:

    .. code-block:: text

        pp:
          tasks: [overlap_analysis, drugtarget]

    and it will skip the other tasks.

* :obj:`consensus: top_down/bottom_up`:
    This key regulates the application of the consensus score. :obj:`top_down` means that it will start at bin 10 (unanimous decision) and go down until it finds the first bin that is not significantly enriched for the positive holdout set. 
    The consensus score will be the last bin that is still significant. :obj:`bottom_up` means that it will start at bin 1 and go up until it reaches the first bin that is significant. This bin will be the consensus score.

* :obj:`cutoff_value: (float)/(int)`: 
    A number between 0 and 1, indicating the cut-off of uncalibrated probabilities assigned to the genes. 
    In other words, setting it to 0.7 means that all genes with a prediction higher than 0.7 will receive a vote from this model. 
    Increasing the cutoff value decreases the number of genes that are voted to be disease genes.
    Can also be an integer above 1, in case :obj:`cutoff_type`: is set to :obj:`top/bottom`.

* :obj:`cutoff_type: split/top/bottom`: 
    indicating the way that the cutoff is applied. :obj:`split` means that the predictions are split at the given float, as explained right above. 
    :obj:`top` means that the top :obj:`k` genes are selected, but the the :obj:`cutoff_value` has to be an integer (i.e. 100 as in Schulte-Sasse et al.)
    :obj:`bottom`means that the bottom :obj:`k` genes are ignored and the rest is selected.

* :obj:`save: True`
* :obj:`save_dir: ./results/`
* :obj:`plot: True`
* :obj:`plot_dir: ./plots/`
    These keys are pretty self-explanatory. So, if you have lost your results or ask yourself where the plots went, this is where you should look.

Now, if you think that you want to try some changes in these settings, i.e. a different :obj:`cutoff_value` or different :obj:`tasks`, just add the respective lines to your :obj:`cardiovascular_gcn.yaml` and re-run the postprocessing only:

Have a little fun with it, but don't forget that hyperparameter hacking and repeated hypothesis testing with different settings like this weakens your evidence!

