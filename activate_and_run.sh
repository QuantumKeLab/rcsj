#!/bin/zsh
source ~/.zshrc
echo "Activating conda environment..."
conda activate py3.11
echo "Running plottr-inspectr..."
plottr-inspectr
echo "Script completed."
