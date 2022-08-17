#!/bin/bash

grep "GcG" hetionet-v1.0-edges.tsv | awk -v FS="\t" '{print $1, $3}' | sed -e 's/Gene:://g' > hetionet_covaries.tsv 