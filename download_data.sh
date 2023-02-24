#!/bin/bash

# Get files from Google Drive

# $1 = file ID
# $2 = file name

fileid="1OPuX8pQRZZ3KGwI_l9_asUL5FVLfr1ex"
filename="data.tar.gz"

URL="https://docs.google.com/uc?export=download&id=$fileid"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt

tar xzvf data.tar.gz