#!/usr/bin/env bash
# Download the Zenodo 10393540 active-layer calorimeter dataset.
#
# Run this ON THE CLUSTER (not your laptop) — these files are large and
# training happens on the cluster anyway. The laptop only needs a small
# dev sample, which you can scp back manually if you want one.
#
# Usage:
#   bash scripts/download_data.sh                     # default DATA_DIR=./data/ad_504
#   DATA_DIR=/scratch/$USER/calo_ad_data bash scripts/download_data.sh
#
# Optional: pass --signals to also fetch the signal zip.
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data/ad_504}"
BASE_URL="https://zenodo.org/records/10393540/files"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

fetch() {
    local name="$1"
    if [[ -f "$name" ]]; then
        echo "[skip] $name already exists ($(du -h "$name" | cut -f1))"
        return
    fi
    echo "[fetch] $name"
    if command -v wget >/dev/null 2>&1; then
        wget --continue --tries=3 "$BASE_URL/$name"
    else
        curl -L --fail --retry 3 -C - -o "$name" "$BASE_URL/$name"
    fi
}

fetch gamma_1.hdf5
fetch gamma_2.hdf5

if [[ "${1-}" == "--signals" ]]; then
    fetch files_fixed_disp.zip
    if [[ ! -d files_fixed_disp ]]; then
        echo "[unzip] files_fixed_disp.zip"
        unzip -q files_fixed_disp.zip -d files_fixed_disp
    fi
fi

echo
echo "Done. Contents of $DATA_DIR:"
ls -lh
