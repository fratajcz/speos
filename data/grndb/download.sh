#!/bin/bash

while read tissue
do
  wget -O ${tissue,,}.txt grndb.com/download/txt?condition=${tissue}_GTEx
done <tissues.txt