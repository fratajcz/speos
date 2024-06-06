def preprocess_mydata(path):
    import pandas as pd

    df = pd.read_csv(path, sep="\t", header=0, index_col =0)

    return df


def test_preprocess_labels(path) -> set:
    import pandas as pd

    return set(pd.read_csv(path, sep="\t", header=None, names=["0"])["0"].tolist())


def preprocess_labels(path) -> set:
    import pandas as pd

    return set(pd.read_csv(path, sep="\t", header=0)["HGNC"].tolist())


def preprocess_labels_mendelian(path) -> set:
    import pandas as pd
    known_positives_df = pd.read_csv(path, sep="\t", names=["chromosome", "start", "end", "symbol", "strand"])
    known_positives_set = set(known_positives_df["symbol"].tolist())

    return known_positives_set