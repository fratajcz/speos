#!/bin/bash

# Get files from Google Drive

wget -O data.tar.gz  "https://drive.usercontent.google.com/download?id=1OPuX8pQRZZ3KGwI_l9_asUL5FVLfr1ex&export=download&confirm=yes"

tar xzvf data.tar.gz
