import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import cm

parser = argparse.ArgumentParser(description='Split a finished testbench run into performance on connected vs disconnected')

parser.add_argument('--disorder', "-d", type=str,
                    help='Path to the config that should be used for the run.')
parser.add_argument('--column', "-c", type=str,
                    help='Path to the parameters that should be used for the run.')
parser.add_argument('--row', "-r", type=str,
                    help='Path to the parameters that should be used for the run.')
parser.add_argument('--threshold', "-t", type=int, default=1,
                    help='Path to the parameters that should be used for the run.')

args = parser.parse_args()

viridis = cm.get_cmap('viridis')

pretty_names = {"film": "Breadth",
                "tag": "Depth",
                "gcn": "GCN"}

results = []
disorder = args.disorder
method = args.row
with open("/lustre/groups/epigenereg01/projects/ppi-florin/results/{}_{}outer_results.json".format(disorder, method.lower())) as file:
    results.append(set([key for key, value in json.load(file)[0].items() if value >= args.threshold]))

n_reps = 3
for i in range(1, n_reps):
    try:
        with open("/lustre/groups/epigenereg01/projects/ppi-florin/results/{}_{}_rep{}outer_results.json".format(disorder, method.lower(), i)) as file:
            results.append(set([key for key, value in json.load(file)[0].items() if value >= args.threshold]))
    except FileNotFoundError:
        continue


other_results = []

method = args.column
with open("/lustre/groups/epigenereg01/projects/ppi-florin/results/{}_{}outer_results.json".format(disorder, method.lower())) as file:
        other_results.append(set([key for key, value in json.load(file)[0].items() if value >= args.threshold]))

n_reps = 3
for i in range(1, n_reps):
    try:
        with open("/lustre/groups/epigenereg01/projects/ppi-florin/results/{}_{}_rep{}outer_results.json".format(disorder, method.lower(), i)) as file:
            other_results.append(set([key for key, value in json.load(file)[0].items() if value >= args.threshold]))
    except FileNotFoundError:
        continue

overlaps = []

total_result = set()
for row, row_result in enumerate(results):
    for col, col_result in enumerate(other_results):
        overlap = len(row_result.intersection(col_result)) / min(len(row_result), len(col_result))
        overlaps.append(overlap)

final = np.asarray(overlaps).reshape((min(len(results), len(other_results)),min(len(results), len(other_results))))

print(final)

fig, ax = plt.subplots()
positions = list(range(1, 1 + n_reps))
labels = ["{} {}".format(pretty_names[args.column], rep) for rep in range(n_reps)]

ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
ax.xaxis.set_major_formatter(ticker.FixedFormatter([""] + labels))

labels = ["{} {}".format(pretty_names[args.row], rep) for rep in range(n_reps)]
ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
ax.yaxis.set_major_formatter(ticker.FixedFormatter([""] + labels))

cax = plt.matshow(final, 0, cmap=viridis, vmin=0, vmax=1)

for (i, j), z in np.ndenumerate(final):
    ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

fig.colorbar(cax)
fig.suptitle("Overlap for multiple runs of {}".format(args.disorder.capitalize()))
fig.tight_layout()
fig.savefig("Overlap_{}_{}_{}.svg".format(args.disorder, args.row, args.column), bbox_inches="tight")