External Validation
===================

Doing machine learning with relatively scarce labeled data points is always challenging to validate, especially since Speos is designed for positive-unlabeled scenarios, where we assume that some of the unlabeled genes are actually positives, 
making the 'internal' validation with a hold-out set somewhat unreliable. To improve upon this weakness, we have added an array of external validation datasets which serve as alternative label sets. The datasets have been selected to have the lowest possible bias, i.e. not being influenced by the training labels.

The external validations are run by the :obj:`speos.postprocessing.postprocessor.Postprocessor` class, which is automatically run when running the :obj:`outer_crossval.py` and :obj:`postprocessing.py` pipelines, as `detailed here <https://speos.readthedocs.io/en/latest/api.html#post-processing-in-detail>`_. 
Before the postprocessor can perform the external validation, you have to train a crossvalidation ensemble, read `here <https://speos.readthedocs.io/en/latest/api.html#the-nested-crossvalidation>`_ on how to do this if you haven't done it already.

Now, we want to take a look into the individual means of external validation, or tasks, how they are called within the framework. To do that, we will look at the log of a run that produced candidate genes for cardiovascular disease. If you cant find your log, check your config file, the logs are placed in :obj:`<config.logging.dir>/<config.name>`.

Differential Gene Expression
----------------------------

The DGE task relies on data obtained from the GEMMA database. We have defined several sub-phenotypes for every disorder and queried GEMMA for genes that are differentially expressed if that sub-phenotype is present. For further methodological details on this task consult the method section in our `preprint <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1.full.pdf>`_ 

One of the subtypes defined for the disorder cardiovascular disease is coronary artery disease. The related part of the logfile is as follows:

.. code-block:: text
    :linenos:

    cardiovascular_gcn 2023-02-22 14:48:35,484 [INFO] speos.postprocessing.postprocessor: Starting Differential Gene Expression Enrichment Analysis.
    cardiovascular_gcn 2023-02-22 14:48:35,523 [INFO] speos.postprocessing.postprocessor: Found 6 subtypes for phenotype cardiovascular_disease: ['Coronary Artery Disease', 'Atrial Fibrillation', 'Aortic Aneurysm', 'Ischemia', 'Hypertension', 'Atherosclerosis'].
    cardiovascular_gcn 2023-02-22 14:48:35,691 [INFO] speos.postprocessing.postprocessor: Total of 552 Coronary Artery Disease DE genes, 473 of them match with our translation table.
    cardiovascular_gcn 2023-02-22 14:48:35,691 [INFO] speos.postprocessing.postprocessor: Found 98 Coronary Artery Disease DE genes among the 584 known positive genes (p: 7.96e-50, OR: 8.798), leaving 375 in 16736 Unknowns
    cardiovascular_gcn 2023-02-22 14:48:35,694 [INFO] speos.postprocessing.postprocessor: Fishers Exact Test for Coronary Artery Disease DE genes among Predicted Genes. p: 4.49e-21, OR: 4.674
    cardiovascular_gcn 2023-02-22 14:48:35,694 [INFO] speos.postprocessing.postprocessor: Coronary Artery Disease DE genes Confusion Matrix:
    [[   66   309]
    [  715 15646]]

This indicates that, while in total 552 genes are labeled as differentially expressed, only 473 match with the HGNC symbols that are contained in our graph. 

Second, 98 of the 473 DGE genes can be found within the 584 Mendelian disorder genes, which corresponds to an odds ratio (OR) of 8.798 with a p-value of 7.96e-50. Quite significant! This significant odds ratio serves as positive control, meaning that the differentially expressed genes for coronary artery disease are indead related to our label set for cardiovascular disease. This leaves 473 - 98 = 375 in the total 16736 unlabeled genes from which we predict our candidates.

Third, when looking at the confusion matrix, 66 out of 781 (66 + 715) candidates are differentially expressed, which corresponds to an OR of 4.674 with a p-value of 4.49e-21. While this is not as high as for the the Mendelian disorder genes, it is still quite high!

This is now done for the other 5 registered sets of differentially expressed genes.

TODO: document other tasks