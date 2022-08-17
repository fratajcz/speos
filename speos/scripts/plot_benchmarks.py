def plot_results(methods, paths, x_label, x_tick_labels, rotate_x_tick_labels: bool = False, num_folds: int = 4, num_repetitions: int = 4, num_metrics = None):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import matplotlib.font_manager
    from matplotlib import rc
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=False)

    colors = ["orange",
              "blue",
              "green",
              "purple",
              "red",
              "brown",
              "pink",
              "grey"]

    abbreviations = {"mrr": "MRR",
                     "auroc": "AUROC",
                     "auprc": "AUPRC"}

    if num_metrics is None:
        df = pd.read_csv(paths[0], sep="\t", header=0, index_col=0)
        num_metrics = len(df.columns)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 12), sharex=False)

    used_methods = []

    patches = {}

    for k, df_path in enumerate(df_paths):
        step = num_folds * num_repetitions
        try:
            df = pd.read_csv(df_path, sep="\t", header=0, index_col=0)
            used_methods.append(methods[k])
        except FileNotFoundError:
            print("Did not find File: {}".format(df_path))
            continue
        means = np.empty((len(df.index) // step, len(df.columns)))
        sds = np.empty((len(df.index) // step, len(df.columns)))
        maxes = np.empty((len(df.index) // step, len(df.columns)))
        mins = np.empty((len(df.index) // step, len(df.columns)))
    
        for i, start in enumerate(range(0, len(df.index), step)):
            one_run = df[start:(start + step)]
            means[i,:] = one_run.mean(axis=0)
            sds[i,:] = np.mean([one_run.iloc[j::4].std(axis=0) for j in range(num_folds)], axis = 0)
            maxes[i,:] = one_run.max(axis=0)
            mins[i,:] = one_run.min(axis=0)

        for j, ax in enumerate(axes):
            try:
                patch, = ax.plot(list(range(len(adjs))), means[:,j], color=colors[k])
            except ValueError:
                patch, = ax.plot(list(range(len(adjs)))[1:], means[:,j], color=colors[k])
            patches.update({k: patch})
            try:
                ax.fill_between(list(range(len(adjs))), means[:,j]-sds[:,j], means[:,j]+sds[:,j], color=colors[k], alpha=0.1, zorder=-1)
            except ValueError:
                ax.fill_between(list(range(len(adjs))[1:]), means[:,j]-sds[:,j], means[:,j]+sds[:,j], color=colors[k], alpha=0.1, zorder=-1)
            #ax.fill_between(list(range(len(adjs))), mins[:,j], maxes[:,j], color=colors[k], alpha=0.1, zorder=-1)
            ax.set_ylabel(" ".join([word.capitalize() if word not in abbreviations.keys() else abbreviations[word] for word in df.columns[j].split("_")]))
            ax.grid(True)
            ax.set_xticks(list(range(len(adjs))), adjs)
            ax.set_xlabel(x_label)

    if rotate_x_tick_labels:
        fig.autofmt_xdate(rotation=90)
    plt.legend(list(patches.values()), used_methods)
    plt.tight_layout()

    return fig, axes


import matplotlib.pyplot as plt

methods = ["cardiovascular","immune"]

#df_paths = ["../nhid_{}_benchmark_nhid_new.tsv".format(method) for method in methods]
df_paths = ["grn_{}_benchmark_grn.tsv".format(method) for method in methods]
adjs = ["MLP","All","PPI","GRN","Adip.T.","Adr.Gland","Blood","BloodVessel","Brain", "Breast","Colon","Esoph.","Heart","Kidney","Liver","Lung","Muscle","Nerve","Ovary","Pancreas","Pituitary","Prostate","Saliv.Gland","Skin","Sm.Int","Spleen","Stomach","Testis","Thyroid","Uterus","Vagina"]

fig, axes = plot_results(methods, df_paths, x_label='Adjacency', x_tick_labels=adjs, rotate_x_tick_labels=True)
plt.savefig("test_benchmark_plot.png", dpi=450)