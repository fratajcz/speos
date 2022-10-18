def preprocess_mydata(path):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=0, index_col =0)

    return df

def preprocess_pancreatitis(path):
    import pandas as pd

    values = []
    hgncs = []
    with open(path, "r") as file:
        for i, line in enumerate(file):
            if i == 0:
                continue

            split_line = line.split("\t")
            hgnc = split_line[0]
            
            if hgnc in hgncs:
                continue

            hgncs.append(split_line[0])
            values.append(split_line[17])
    df = pd.DataFrame(data=values, index=hgncs, columns=["P_DOM"])
    df = df[df["P_DOM"] != "NA"]
    df.index.name = "hgnc"
    #df = pd.read_csv(path, sep="\t", header=0, index_col=False)

    return df