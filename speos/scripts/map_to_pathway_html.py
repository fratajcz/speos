# this script requres the additional installation of html5lib and bs4

import requests
from bs4 import BeautifulSoup
import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(description='''Takes the results of a Wikipathways GSEA, downloads the path html and highligts relevant nodes.
                                              example usage: python speos/scripts/map_to_pathway_html.py -i rescued/gsea/immune_dysregulation_film_pathwayea.tsv -o WP5130 -m /mnt/storage/speos/data/mendelian_gene_sets/Immune_Dysregulation_genes.bed''')

parser.add_argument('--input', "-i", type=str, help="Path to the resulting dataframe of the WikiPathways GSEA done by the postprocessor")
parser.add_argument('--only', "-o", type=str, default="", help="Limit the analysis to one path (by Wikipathways ID, i.e. WP1234)")
parser.add_argument('--top', "-t", type=int, default=-1, help="Limit the analysis to the top n paths by p-value")
parser.add_argument('--mendelian', "-m", type=str, default="", help="Path to a Mendelian Gene file that will be used for additional highlighting")

args = parser.parse_args()

def get_translation_table(path="/mnt/storage/speos/data/hgnc_official_list.tsv", sep="\t") -> pd.DataFrame:
    print("Reading translation table from {}".format(path))
    df = pd.read_csv(path, sep=sep, header=0, usecols=["symbol", "entrez_id", "ensembl_gene_id"])
    return df


df = pd.read_csv(args.input, header=0, index_col=0, sep="\t")

if args.top != -1:
    df = df[:args.top, :]

if args.only != "":
    URLs = [(args.only, "https://pathway-viewer.toolforge.org/embed/{}".format(args.only))]
else:
    URLs = [(index, "https://pathway-viewer.toolforge.org/embed/{}".format(index)) for index in df.index]

if args.mendelian != "":
    mendelian_genes = pd.read_csv(args.mendelian, sep="\t", names=["chr", "start", "stop", "gene", "strang"], usecols=["gene"])["gene"].tolist()
    translation_table = get_translation_table()
    entrez2symbol = {line[1]: line[0] for i, line in translation_table.iterrows()}

    path = "/mnt/storage/speos/data/pathways/wikipathways-20220710-gmt-Homo_sapiens.gmt"

    pathway2symbol = {}
    pathway2name = {}

    with open(path, "r") as file:
        for line in file.readlines():
            fields = line.split("\t")
            pathway = fields[0].split("%")[2]
            name = fields[0].split("%")[0]
            entrez_ids = [int(id.strip()) for id in fields[2:]]
            pathway2symbol.update({pathway: [entrez2symbol[entrez_id] for entrez_id in entrez_ids if entrez_id in entrez2symbol.keys()]})

for index, URL in URLs:
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html5lib')

    candidate_genes = df.loc[index, "genes"].split(";")

    for indicator_color, geneset, genes in zip(["#FF0000", "#00FF00"], ("Mendelians", "Candidates"), [mendelian_genes, candidate_genes]):

        # first filter the Mendelians for genes that actually occur in the pathway to prevent spurious results
        if geneset == "Mendelian":
            genes = [gene for gene in genes if gene in pathway2symbol[index]]

        for gene in genes:
            # first with an exact name match
            tags = soup.find_all('a', attrs = {'name' : gene})
            if tags is None or len(tags) == 0:
                tags = soup.find_all('g', attrs = {'name' : gene})

            # if that doesnt work out, search fur fitting substrings in the class
            if tags is None or len(tags) == 0:
                tags = soup.find_all('a', class_=re.compile(" {} ".format(gene)))
            if tags is None or len(tags) == 0:
                tags = soup.find_all('g', class_=re.compile(" {} ".format(gene)))

            if tags is None or len(tags) == 0:
                assert geneset == "Mendelians"
                continue
            
            if tags[0]["name"] != gene:
                print("Mapped {} to {}".format(gene, tags[0]["name"]))

            for tag in tags:
                text = tag.findChildren("text" , recursive=False)[0]
                if text["fill"] == "#000000":
                    text["fill"] = indicator_color
                elif text["fill"] == "#FF0000" or text["fill"] == "#00FF00":
                    text["fill"] = "#FFFF00"
                else:
                    text["fill"] = "#0000FF"

                text = tag.findChildren("rect" , recursive=False)[0]
                text["stroke"] = "#FF0000"

    with open("{}_mapped.html".format(index), "w") as file:
        file.write(str(soup.prettify()))