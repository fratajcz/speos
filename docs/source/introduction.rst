Introduction
============

What is Speos about?
--------------------

Recently, `Boyle et al (2017) <https://pubmed.ncbi.nlm.nih.gov/28622505/>`_ have proposed the omnigenic model to structure the genetic influence on any trait. They postulate that core genes influence the phenotype directly by their protein expression, while others, the peripheral genes, influence the phenotype indirectly by influencing the core genes.

Several works have `expanded <https://pubmed.ncbi.nlm.nih.gov/31051098/>`_ this idea or `argued <https://pubmed.ncbi.nlm.nih.gov/29906445/>`_ against it, which is why we won't do that here.

The bottom line is that under the assumption of the omnigenic model, knowing all core genes that influence a trait, or say, a disease, gives us an idea about which proteins are most likely involved in the disease etiology and also pointers where to look for treatment.

It is hard to make a claim that a gene is a core gene, except for rare examples of Mendelian disorder genes, which obviously have a direct and sizable contribution to the disease. In other cases, gene expression patterns as well as GWAS signal is expected to be relevant. Furthermore, the genetic effects are expected to propagate through networks of biological modulation to exert their effects on the phenotype.

We have therefore developed Speos to find yet undiscovered core genes for common complex diseases using rare Mendelian disorder genes as "true positive" core genes during training.
As input data we have implemented tissue-specific gene expression and GWAS z-scores as well as several types of biological networks. 

`We have shown <https://www.biorxiv.org/content/10.1101/2023.01.13.523556v1>`_ that by utilizing these ingredients carefully, Speos enables the prediction of promising core gene candidates for five common complex diseases.

As we know, some diseases are more relevant to some researchers than others. Also, several disease have specific, disease-relevant input feautures that can help identify core genes. 
For this reason, we have made Speos completely extensible, from the set of core genes used for training, to the input features, the networks as well as the actual methods that are used during training!
We would be very happy if Speos can help you on your research journey. If you are interested in using speos but don't know how or don't know you way around a command line interface, get in touch with us via our `GitHub issue page <https://github.com/fratajcz/speos/issues>`_ and we'll find a solution.

What do I need to run Speos?
----------------------------

Apart from compute resources (which highly depend on the model that you choose) you'll need only one thing to use Speos: A labeled set of known (or assumed) core genes for the disease or trait that you are interested in. That's it. The set should consist of at least 100 genes (our smallest set has roughly 120 genes), but can in theory be smaller. If you are uncertain how to obtain such genes, you can check `OMIM <https://www.omim.org/>`_ for genes that produce the phenotype that you are interested in, but there are several other ways to come up with genes for training.

Second, it is great if you can map some GWAS traits to your disease. In our work, we use several GWAS traits related to cardiovascular health, such as HDL, LDL, TRIG etc., and map it to the disease genes for cardiovascular disease. You can check the Extensions page on how to do that, but it is not a must.

Lastly, if you have disease-relevant gene features that you want to use or specific disease-relevant biological networks, you can add them, too, but that is optional as well.

How does Speos work?
--------------------

Underneath, Speos trains several machine learning models in a nested cross-validation and uses their predictions to obtain a consensus score for each gene. Generally speaking, the higher the consensus score, the more likely a gene is to be a core gene.
A consensus score of 0 on the other hand means that the models are fairly certain that the gene is not a core gene. Each gene that has not been used as "true positive" during training receives a consensus score, which is then validated on external datasets to check if the predicted candidate genes share properties that are expected of core genes.

Where can I start?
------------------

This documentation is intended to be read from start to finish and will guide you through the whole process, step by step. Some parts might not apply to you, so feel free to skip them.
On the next page you will be guided through the installation process. Afterwards, you will be guided through the creation of a configuration file and how you can use it to customize your training runs. You will not need to write any code, changing settings in a plain text file is all you need! Finally, once you got the hang of how to run Speos, you can add your own datasets and use them to discover new core genes.
