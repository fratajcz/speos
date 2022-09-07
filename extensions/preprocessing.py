def preprocess_mydata(path):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=0, index_col =0)

    return df