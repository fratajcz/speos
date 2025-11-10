#!/bin/bash

# Get files from Google Drive

# $1 = file ID
# $2 = file name

wget -O data.tar.gz https://zenodo.org/records/17234470/files/data.tar.gz

tar xzvf data.tar.gz
