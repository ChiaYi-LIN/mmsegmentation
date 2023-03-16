#!/bin/bash

root="./data/"
url="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
filename="./data/ADEChallengeData2016.zip"

mkdir -p $root

if [ -f "$filename" ]; then
    echo "$filename already exists, skipping download."
else
    echo "Downloading $filename..."
    wget "$url" -P $root
    echo "Download complete."
fi

unzip -qo $filename -d $root