![SPEOS Header](img/speos_space_11_1080.png "SPEOS")

# Identification of core genes from Biological Networks, GWAS and gene expression.

Speos, pronounced almost like "space", is a machine learning framework to merge deep learning and the omnigenic model. Its goal is to predict core genes for several diseases which allow subsequent research to allocate resources to the most promising candidates.

# Use of this branch

This branch contains the analyses accompanying the manuscript "Exploring the Omnigenic Architecture of Common Complex Traits". It is based on Speos `latest` branch, so if you want to install Speos to explore the omnigenic architecture of common complex traits yourself, first go to its [documentation](https://speos.readthedocs.io/en/latest/index.html) for instructions on how to install Speos.

First, you will need to train a Speos model. You can use the configs in `arch_configs` and training genes in `extensions` or add your own, as detailed in the respective part of the [documentation](https://speos.readthedocs.io/en/latest/index.html).

Second, or if you only want to redo our analyses with a gene set you have obtained from other sources, we have broken the analyses from the manuscript down into steps which can be execuded from jupyter notebooks.
This will run all included analyses and generate the same figures as in the manuscript. Note that some analyses, such as inspecting input feature importance, will only be possible if you trained your own Speos model.

Note that reproducing the `GEARS` simulations also requires you to install GEARS, which you can do according to their [documentation](https://github.com/snap-stanford/GEARS).

The following notebooks contain the necessary steps:

- `notebooks/plot_mko.ipynb`
  - This notebook contains the analyses to test whether the your core genes are significantly enriched in mouse knockout and differentially expressed genes. It will also generate figure 1B.
- `notebooks/zscore_distribution.ipynb`
  - This notebook contains the analyses to test whether the Zscores of your core genes are significantly different from the peripheral genes. It will also generate figures 1C and D.
- `notebooks/plot_explanation.ipynb`
  - This notebook aggregates and plots the feature importance scores for the genes of your choice. It will also generate figure 2. To obtain pre-computed feature importance vectors for the results in the manuscript, go to [Zenodo](https://zenodo.org/records/14035135).
- `notebooks/expression_signatures.ipynb`
  - This notebook analyses the enrichment or depletion of core genes among genes that are strongly expressed in certain tissues. It will also generate figure 3.
- `notebooks/importance_subnetwork`
  - This notebook aggregates and plots the edge importance scores and create the importance-weighted subnetworks. It will also generate figure 4A and B. To obtain pre-computed edge importance vectors for the results in the manuscript, go to [Zenodo](https://zenodo.org/records/14035135).
-  `notebooks/goea.ipynb`
  - This notebook contains the pathway and gene ontology enrichment analyses. It will also generate figures 5A and B.
- `notebooks/read_cmap_2017.ipynb`
  - This notebook contains all analyses regarding the CMAP perturbation data. It will also generate figures 6, 7 and 8A, B and C.
- `notebooks/tss_analysis_official.ipynb`
  - This notebook contains the analyses for the regulatory enrichment of core genes. It will also generate figures 8D and E.
- `speos/scripts/gears/plot_genetic_interactions.ipynb`
  - This notebook contains the analyses for the enrichment of core gene pairs among simulated strong interactions. It will also genetate Figures 9D-K. The table that is used to recreate our exact results is available on [Zenodo](https://zenodo.org/records/14035135).
 
Additionally, some scripts might be necessary if you wish to create all the data yourself instead of only recreating it from our data.

- `speos/scripts/gears/train_gears.py`
  - You can re-train the GEARS model if you like. However, we provide the weights of the model we trained on zenodo, which you can use instead.
- `speos/scripts/gears/permute_everything.py`
  - Query your GEARS model to simulate co-perturbations of genes and write them to disk. This data can then be used to run the `scripts/gears/plot_genetic_interactions.ipynb` notebook, if you dont want to use our pre-compiled table.
- `speos/scripts/explanation_scripts/explanation_ensemble_homogeneous.py`
  - Since computing feature and edge importance for all 110 models of the Speos ensemble is computationally intensive, Speos does not do this by default. Therefore, execute this script to run integrated gradients on both the features and edges. It is recommended to do this either only for individual genes, or supplying chunks of genes to several GPUs on a cluster.
- `speos/scripts/explanation_scripts/explanation_fromlist.py`
  - Some GNN layers, such as the `FiLMConv` layer used in our manuscript, are not available for the new explanation API of pytorch geometric. Therefore, to run feature importance calculations for the `FiLMConv` layer, the user has to switch the `message_passing.py` file in their pytorch geometric installation with the file `speos/scripts/explanation_scripts/message_passing_204.py`. This has been tested mostly with pygeo 2.04, but works well until and including 2.2.X


# Citation

If you use Speos in your work, please cite the paper below. You can use the following information:

```
@article{ratajczak_speos_2023,
	title = {Speos: an ensemble graph representation learning framework to predict core gene candidates for complex diseases},
  	author = {Ratajczak, Florin and Joblin, Mitchell and Hildebrandt, Marcel and Ringsquandl, Martin and Falter-Braun, Pascal and Heinig, Matthias},
	journal = {Nature Communications},
	volume = {14},
	url = {https://www.nature.com/articles/s41467-023-42975-z},
	doi = {10.1038/s41467-023-42975-z},
	year = {2023},
	pages = {7206}
}
```
