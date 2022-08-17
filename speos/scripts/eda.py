from datasets import Preprocessor
import pandas as pd
import numpy as np
import umap
import umap.plot
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from joblib import delayed, Parallel
import json

with open("mapping.json", "r") as file:
  content = file.read()
  liste = json.loads(content)

#liste = [liste[0]]

names = []
z_mendel = {}
z_random = {}
z_mendel_neighbors = {}
z_random_neighbors = {}
pval_nodes = []
pval_neighbors = []

n_processes = 8



def get_z_values(parameters):
  preprocessor = Preprocessor(**parameters)
  preprocessor.get_data()
  z_mendel_list, z_random_list, z_neighbors_mendel_list, z_neighbors_random_list = preprocessor.inspect(plot=False)
  pval_nodes = ttest_ind(z_mendel_list, z_random_list,alternative = "greater")[1]
  pval_neighbors = ttest_ind(z_neighbors_mendel_list, z_neighbors_random_list,alternative = "greater")[1]
  return (z_mendel_list, z_random_list, z_neighbors_mendel_list, z_neighbors_random_list, pval_nodes, pval_neighbors, parameters["name"])

if len(liste) > 1:
  # parallel workflow for longer lists
  results = Parallel(n_jobs=n_processes)(delayed(get_z_values)(parameters) for parameters in liste)

else:
  # iterative workflow for short lists/debugging
  results = []
  for parameters in liste:
    results.append(get_z_values(parameters))

for result in results:
  name = result[6]
  names.append(name)
  z_mendel[name] = result[0]  # z_mendel_list
  z_random[name] = result[1]  # z_random_list
  z_mendel_neighbors[name] = result[2] #  z_neighbors_mendel_list
  z_random_neighbors[name] = result[3] # z_neighbors_random_list
  pval_nodes.append(result[4])
  pval_neighbors.append(result[5])

pvals_adjusted = fdrcorrection(pval_nodes + pval_neighbors)[1]

pval_nodes = pvals_adjusted[:len(pval_nodes)]
pval_neighbors = pvals_adjusted[len(pval_nodes):]
pvals = [pval_nodes, pval_neighbors]

all_values =  [z_mendel_values for y in z_mendel.values() for z_mendel_values in y] + [z_random_values for y in z_random.values() for z_random_values in y]

fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=False)
suptitles = ["Nodes", "Neighborhood"]

for i, dicts in enumerate([[z_mendel,z_random], [z_mendel_neighbors, z_random_neighbors]]):
  mendel, random = dicts
  sns.violinplot(ax = axes[i],
                x=[names for key, value in mendel.items() for names in [key] * len(value)] + [names for key, value in random.items() for names in [key] * len(value)],
                y=[mendel_values for y in mendel.values() for mendel_values in y] + [random_values for y in random.values() for random_values in y], 
                hue=[sample for key, value in mendel.items() for sample in ["mendel"] * len(value)] + [sample for key, value in random.items() for sample in ["random"] * len(value)],
                palette="muted", split=False)

  for pval, x, name in zip(pvals[i],range(len(names)), names):
    if i == 0:
      values = z_mendel[name] + z_random[name]
    else:
      values = z_mendel_neighbors[name] + z_random_neighbors[name]

    axes[i].text(s=round(pval,3), x=x-0.1, y=np.max(values) +1)
  
  axes[i].set_xticklabels(names, rotation=30, ha="right")
  axes[i].set_title(suptitles[i])
  #axes[i].tick_params(axis='x', rotation=90)
plt.tight_layout()
fig.savefig("eda/mendel_all_z.png",dpi=450)
fig.clf()
#mapper = umap.UMAP(n_neighbors=500).fit(dataset.data.x)

#p = umap.plot.points(mapper, labels=dataset.data.y)

#umap.plot.show(p)