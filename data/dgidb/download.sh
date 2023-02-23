#!/bin/bash

wget https://www.dgidb.org/data/monthly_tsvs/2022-Feb/categories.tsv

grep "DRUGGABLE GENOME" categories.tsv > druggable_genome.tsv