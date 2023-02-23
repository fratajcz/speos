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


Gene Set Enrichment Analysis
----------------------------

Gene set enrichment analysis (GSEA) has a long-standing tradition in biology. In its essence, it takes a set of genes (in our case our candidate genes) and compares it to an ontology. This ontology consists of groups of genes that group together by a shared funtion, pathway, cellular component or other categories.
Then, a statistical analysis is conducted how the proposed set of genes is distributed among these ontology groups. Some groups are overrepresented compared to random expectation, and this enrichment is expressed in form of a factor and a p-value.

Finally, if the proposed gene set should capture the characteristics of a given trait or disease, we would expect that the enriched functions, pathways etc. are representative for this disease.

In total, five different types of GSEA are pre-implemented in Speos, differing in the type of ontology they use: Pathways (Wikipathways), Human Phenotype Ontology (HPO), and the Gene Ontology (GO) categories biological process, molecular function and cellular component.
For each of the GSEA runs you get a short overview of how many ontology terms are significantly enriched in the logs (see below). On top of that, the full list of enriched terms together with enrichment factor and p-value are written to a tsv table so they can be examined more closely.
Finally, plots are generated that display the same data as the tables for a faster visual inspection.

.. code-block:: text
    :linenos:

    cardiovascular_gcn 2023-02-22 14:48:36,241 [INFO] speos.postprocessing.postprocessor: Starting Pathway Enrichment Analysis.
    cardiovascular_gcn 2023-02-22 14:48:36,336 [INFO] speos.postprocessing.postprocessor: Reading translation table from ./data/hgnc_official_list.tsv
    cardiovascular_gcn 2023-02-22 14:48:39,990 [INFO] speos.postprocessing.postprocessor: Found 34 significant terms, writing table to ./results/cardiovascular_gcn_pathwayea.tsv
    cardiovascular_gcn 2023-02-22 14:48:40,029 [INFO] speos.postprocessing.postprocessor: Saving plot to ./plots/cardiovascular_gcn_pathwayea.png
    cardiovascular_gcn 2023-02-22 14:48:43,542 [INFO] speos.postprocessing.postprocessor: Starting HPO Enrichment Analysis.
    cardiovascular_gcn 2023-02-22 14:48:43,664 [INFO] speos.postprocessing.postprocessor: Reading translation table from ./data/hgnc_official_list.tsv
    cardiovascular_gcn 2023-02-22 14:49:14,194 [INFO] speos.postprocessing.postprocessor: Found 127 significant terms, writing table to ./results/cardiovascular_gcn_hpoea.tsv
    cardiovascular_gcn 2023-02-22 14:49:14,280 [INFO] speos.postprocessing.postprocessor: Saving plot to ./plots/cardiovascular_gcn_hpoea.png
    cardiovascular_gcn 2023-02-22 14:49:21,906 [INFO] speos.postprocessing.postprocessor: Starting GO Enrichment Analysis.
    cardiovascular_gcn 2023-02-22 14:49:45,849 [INFO] speos.postprocessing.postprocessor: Found 78 significant terms for task biological process, writing table to ./results/cardiovascular_gcn_goea_biological_process.tsv
    cardiovascular_gcn 2023-02-22 14:49:45,865 [INFO] speos.postprocessing.postprocessor: Saving plot to ./plots/cardiovascular_gcn_goea_biological_process.png
    cardiovascular_gcn 2023-02-22 14:50:03,321 [INFO] speos.postprocessing.postprocessor: Found 57 significant terms for task molecular function, writing table to ./results/cardiovascular_gcn_goea_molecular_function.tsv
    cardiovascular_gcn 2023-02-22 14:50:03,371 [INFO] speos.postprocessing.postprocessor: Saving plot to ./plots/cardiovascular_gcn_goea_molecular_function.png
    cardiovascular_gcn 2023-02-22 14:50:20,456 [INFO] speos.postprocessing.postprocessor: Found 75 significant terms for task cellular component, writing table to ./results/cardiovascular_gcn_goea_cellular_component.tsv

As an example, here are the first lines of the GSEA for GO biological process:

.. code-block:: text
    :linenos:
    :caption: ./results/cardiovascular_gcn_goea_biological_process.tsv

                    fdr_q_value             p_value genes           description     observed        total   expected        enrichment      log_q
    GO:0042776      6.419590551030668e-20   5.125830845600981e-24   NDUFAB1;NDUFS5;NDUFB1;ATP5PD;ATP5MG;NDUFB6;NDUFB4;NDUFB10;ATP5F1B;NDUFB7;ATP5PF;ATP5PB;ATP5F1A;ATP5F1D;ATP5F1C;ATP5PO;NDUFA8;ATP5MF;ATP5F1E;ATP5ME;NDUFA6;SDHC;NDUFA13;STOML2   Proton Motive Force-driven Mitochondrial Atp Synthesis  24      34      1.5866395793499044      15.126308654063418      19.19249267085618
    GO:0015986      4.3194905567478847e-14  6.897940844375415e-18   ATP5PD;ATP5MG;ATP5MC1;ATP5F1B;ATP5PF;ATP5PB;ATP5F1A;ATP5F1D;ATP5F1C;ATP5PO;ATP5MC3;ATP5MF;ATP5F1E;ATP5ME;ATP5MC2;ATP5MK Proton Motive Force-driven Atp Synthesis        16      20      0.9333173996175909      17.14314980793854       13.36456747111097
    GO:0002181      3.633652945419001e-13   8.704055282862506e-17   RPLP1;RPLP2;RPLP0;RPL35A;RPL13;RPL12;RPL9;RPL4;RPL5;RPL21;RPS5;RPL29;RPL14;RPS3A;RPL26;RPL27;RPS16;RPS29;RPL23;RPS24;RPS25;RPS26;RPL30;RPL10A;RPL32;RPL11;RPL24;RPL19   Cytoplasmic Translation 28      80      3.7332695984703634      7.500128040973111       12.43965655503765
    GO:0006953      6.501057068304837e-13   2.0763516666575653e-16  ASS1;SERPINA1;SERPINA3;A2M;CRP;APCS;FN1;ORM1;AHSG;TFRC;SERPINF2;SAA1;SAA2;LBP;ORM2;SAA4;ITIH4;CD163     Acute-phase Response    18      30      1.3999760994263863      12.857362355953905      12.187016021571488
    GO:0006936      5.727016931615222e-12   2.2864168522896928e-15  FXYD1;MYL1;CKMT2;HRC;CALD1;TRDN;GAMT;TRIM63;MYLPF;ANKRD2;MYH2;CERT1;TMOD4;LMOD2;MYH1;TPM2;TNNT1;TMOD1;MYOM3;TPM4;LMOD1;MYOM1;MYOM2;TNNI1        Muscle Contraction      24      64      2.9866156787762907      8.03585147247119        11.24207153291667
    GO:0006412      4.952007184843684e-10   2.372408424549833e-13   RPLP1;RPLP2;RPLP0;RPL35A;RPL13;RPL12;RPL9;RPL4;RPL5;RPL21;RPS5;RPL29;RPL14;RPS3A;RPL26;RPL27;RPS16;RPS29;RPL23;RPS24;RPS25;RPS26;RPL30;RPL10A;RPL32;RPL11;RPL24;RPL19;EIF4G1;PABPC4;MRPL51;RPL36AL;EEF1A2;MRPL12        Translation     34      154     7.186543977055449       4.731064070372649       9.305218733871682
    GO:0009060      9.349694705770931e-10   5.225795507856637e-13   NDUFAB1;NDUFS5;NDUFB1;NDUFB6;NDUFB4;NDUFB10;UQCRH;NDUFB7;UQCRC2;ATP5F1D;UQCRC1;MDH2;NDUFA8;NDUFA6;OXA1L;SDHC;NDUFA13    Aerobic Respiration     17      37      1.7266371892925432      9.845727930234972       9.029202569850872
    GO:0006958      3.2020976539954114e-08  2.0454153011788e-11     MASP2;C5;C9;C4BPA;SERPING1;CFI;C2;C8A;C8B;C8G;C1S;C7;C1QBP      Complement Activation, Classical Pathway        13      24      1.119980879541109       11.60734101579172       7.4945654275870375
    GO:0045214      2.846309025773699e-08   2.0454153011788e-11     KLHL41;ITGB1;CAPN3;CASQ1;MYOM2;ANKRD1;LMOD2;SYNPO2L;MYOZ1;CFL2;CSRP1;TNNT1;WDR1 Sarcomere Organization  13      24      1.119980879541109       11.60734101579172       7.545717950034419
    GO:0045333      3.284414314881856e-08   2.6224962590880362e-11  NDUFA4;UQCRQ;UQCR11;UQCRH;CYC1;COX6C;COX5B;COX4I1;COX7C;COX5A;UQCRC2;UQCRC1;CYCS;UQCR10 Cellular Respiration    14      29      1.3533102294455068      10.34500419444567       7.483542063697089
    GO:0030239      3.098886957011646e-08   2.721794676391577e-11   KLHL41;CAPN3;MYOZ1;MYL9;PGM5;LMOD2;FLII;TMOD4;LMOD1;TMOD1       Myofibril Assembly      10      13      0.606656309751434       16.483797892248596      7.508794265916731
    GO:0006956      2.9778377494698357e-07  2.8532460071573e-10     CFD;C2;C8A;CFHR1;CFHR3;C7;CFB;C9;CFHR2;CFHR4;C8B        Complement Activation   11      19      0.8866515296367113      12.406226834692363      6.526098968917478
    GO:0006957      3.5807728270686596e-07  3.7168673548301316e-10  CFD;CFB;C5;C9;C8A;C8B;C8G;C7    Complement Activation, Alternative Pathway      8       9       0.41999282982791586     19.047944231042823      6.446023230811629

And here is the accompanying plot, truncated to the top 10:

.. image:: https://raw.githubusercontent.com/fratajcz/speos/master/docs/img/cardiovascular_gcn_goea_biological_process_top10.png
  :width: 600
  :alt: Benchmark Results


TODO: document other tasks