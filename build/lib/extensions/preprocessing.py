def preprocess_mydata(path):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=0, index_col =0)

    return df

def test_preprocess_labels(path) -> set:
    import pandas as pd

    return set(pd.read_csv(path, sep="\t", header=None, names=["0"])["0"].tolist())
