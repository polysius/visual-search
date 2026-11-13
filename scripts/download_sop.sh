#!/usr/bin/env bash
# Download and unpack Stanford Online Products (about 2.9 GB) into DATA_DIR.
# Usage: DATA_DIR=~/vs-data scripts/download_sop.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:?set DATA_DIR to the folder that should hold the dataset}"
URL="ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"

mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/Stanford_Online_Products/Ebay_test.txt" ]; then
    curl --fail --continue-at - --output "$DATA_DIR/sop.zip" "$URL"
    unzip -q "$DATA_DIR/sop.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/sop.zip"
fi
echo "Dataset ready in $DATA_DIR/Stanford_Online_Products"
