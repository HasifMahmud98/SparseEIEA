#!/bin/bash
for dataset in FBDB15K FBYG15K zh_en ja_en fr_en; do
    echo "========================================="
    echo "Running on $dataset"
    echo "========================================="
    python -c "from main_real import run_real_experiment; run_real_experiment('./data/mmkg', '$dataset')"
done