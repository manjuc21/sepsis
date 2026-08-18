#!/usr/bin/env bash
# Downloads the PhysioNet/CinC 2019 Early Prediction of Sepsis Challenge dataset
# into data/raw/{training_setA,training_setB}/. ~40,000 per-patient PSV files.
#
# Source: https://physionet.org/content/challenge-2019/1.0.0/
# No login/data-use-agreement required (public, de-identified).

set -euo pipefail

BASE_URL="https://physionet.org/files/challenge-2019/1.0.0/training"
RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
PARALLELISM="${PARALLELISM:-32}"

mkdir -p "$RAW_DIR"

download_set() {
  local set_name="$1"
  local dest="$RAW_DIR/$set_name"
  mkdir -p "$dest"

  echo "Fetching file list for $set_name..."
  local listing
  listing=$(curl -sf --max-time 30 "$BASE_URL/$set_name/")
  echo "$listing" | grep -oE 'href="p[0-9]+\.psv"' | sed -E 's/href="(.*)"/\1/' > "$dest/.filelist"

  local total
  total=$(wc -l < "$dest/.filelist")
  echo "Downloading $total files for $set_name with $PARALLELISM parallel workers..."

  awk -v dest="$dest" -v base="$BASE_URL/$set_name" '{print base"/"$1" -> "dest"/"$1}' "$dest/.filelist" > /dev/null

  cat "$dest/.filelist" | xargs -P "$PARALLELISM" -I{} \
    curl -sf --max-time 20 -o "$dest/{}" "$BASE_URL/$set_name/{}" \
    || echo "warning: some files in $set_name may have failed, re-run to retry missing ones"

  rm -f "$dest/.filelist"
  echo "$set_name done: $(ls "$dest" | wc -l) files in $dest"
}

download_set "training_setA"
download_set "training_setB"

echo "Download complete."
echo "training_setA: $(ls "$RAW_DIR/training_setA" | wc -l) files"
echo "training_setB: $(ls "$RAW_DIR/training_setB" | wc -l) files"
