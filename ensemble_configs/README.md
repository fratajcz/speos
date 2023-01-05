# Ensemble Configs

This configs can be used to train ensembles and obtain candidate gene predictions. The file name identifies the disease and method that is used by the ensemble.

To train an ensemble, install Speos and run the following line:

```
python crossval.py -c <path-to-config>
```

Be aware that this will preprocess the data, train 110 models, assess their overlaps and do the external validations. With a state of the art GPU this can take up to 4 hours, depending on the GPU and the selected method. Without a GPU, the training is expected to take substantially longer.
