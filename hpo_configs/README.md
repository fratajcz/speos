# HPO Configs

This configs can be used to recreate the hpo and benchmark runs in the manuscript. 

To run an array of benchmarks, install Speos and run the following line:

```
python run_benchmark.py -c <path-to-config> -p <path-to-parameters>
```

The configs are identifiable by the wird `config` in the file name, while the parameter files have the word `parameters` in them. Broadly speaking, the configs define the shared characteristics across all runs while the parameter files define the individual hpo parameters.

For example, running

```
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_immu.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_gcn_new.yaml
```

will produce the runs for the red boxplots in Figure 1b, while the following line produces the green boxplots:

```
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_immu.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_tag_new.yaml
```

(the parameter files also contain the settings for MLP, RGCN and FiLM runs)

Now, to do both for cardiovascular disease isntead of immune dysregulation, just swap the config file:


```
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_cardio.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_gcn_new.yaml
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_cardio.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_tag_new.yaml
```

The additional settings with skip-connections and concatenations (Extended Data Figure 6a) can be obtained with the respective config files, too:

```
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_immu_skip.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_tag_new.yaml
python run_benchmark.py -c hpo_configs/adjacencies/testbench_config_adj_immu_concat.yaml  -p hpo_configs/adjacencies/testbench_adjacencies_parameters_tag_new.yaml
```

In general, the following subdirectoreis contain configs and parameter files for the following HPO details:

adjacencies: Comparison of all adjacencies, Figure 1b, Extended Data Figure 6a
gnn_layers: Comparison of different GNN layers and number of GNN layers, Extended Data Figure 5
benchmark: Additional Benchmark settings necessary to put together Figure 1a
nhid: Number of hidden dimensions, not shown in manuscript
