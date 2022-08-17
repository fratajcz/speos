import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud

import matplotlib.pyplot as plt


#df = pd.read_csv("/home/icb/florin.ratajczak/ppi-core-genes/results/goanalysis/cardiovascular_disease_biol_process.txt", index_col=False, skiprows=5,header=6,sep="\t")
#df = pd.read_csv("/home/icb/florin.ratajczak/ppi-core-genes/results/goanalysis/immune_dysreg_biol_process.txt", index_col=False, skiprows=5,header=6,sep="\t")
#df = pd.read_csv("/home/icb/florin.ratajczak/ppi-core-genes/results/goanalysis/diabetes1_biol_process.txt", index_col=False, skiprows=5,header=6,sep="\t")
df = pd.read_csv("/home/icb/florin.ratajczak/ppi-core-genes/results/goanalysis/insulin_disorder_biol_process.txt", index_col=False, skiprows=5,header=6,sep="\t")


#text = df.iloc[:,0][0]

text = " ".join(words for review in df.iloc[:,0] for words in review.split("(")[0].split("\w"))

print(text)
# Create and generate a word cloud image:
wordcloud = WordCloud(background_color="white",height=720,width=1080).generate(text)
wordcloud.to_file("insulin_disorder.png")
