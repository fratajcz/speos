import networkx as nx
import pandas as pd

df = pd.read_table("~/ppi-core-genes/data/drkg/cgi_clean.tsv",sep="\t",names= ["Compound", "edge", "Gene"])

node2entrez = {value: "".join(value.split("::")[1:]) for value in df["Gene"] if not "".join(value.split("::")[1:]).startswith("drugbank")}

node2compound = {value: "".join(value.split("::")[1:]) for value in df["Compound"]}

graph = nx.from_pandas_edgelist(df,source="Compound",target="Gene",edge_attr="edge",create_using=nx.MultiDiGraph)


in_degree = {node2entrez[node]: graph.in_degree[node] for node in node2entrez.keys()}

print(graph.info())