High-Level API
==============

Speos has a high-level API that lets you run jobs, do postprocessing and even hyperparameter search with only a config file and a call to the respective pipeline.
For convenience, we have packaged the pipelines in command-line scripts so that you don't have to get your fingers dirty at all.

Let's first come up with an example config file. The whole list of settings that you can manage and their default values in your config file is shown in the `config_default.yaml <https://github.com/fratajcz/speos/blob/master/speos/utils/config_default.yaml>`_

Lets write a config file like the following:

::
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

>>>python training.py -c my_config.yaml

This will trigger a training run using the Mendelian genes for Cardiovascular Disease as ground truth labels, BioPlex 3.0 293T as adjacency and a vanilla GCN as graph convolution.
The rest of the settings will be default. 

After the model has been trained and early stopped on the holdout set, an Inference will be triggered and predictions for all genes will be produced. The settings for the inference, like cutoff value or save path,
are defined in the default config. 

Inference Only
--------------

Let's say you just ran the training command above but you can't find the directory where the results are saved, or the results have been deleted.
We can modify the config from above to explicitely tell Speos to save inference results to a specific directory by adding the following lines:

::
    inference:
        save_dir: ~/my_results/

and re-run only the inference:

>>>python inference.py -c my_config.yaml

which will save all the results of the inference to :obj:`~/my_results/` without training the model again.
This assumes that the model has not been moved/deleted since it has been trained and that the config settings for the model save path are the same like when it was trained.