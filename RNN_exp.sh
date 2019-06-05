#!/bin/bash

# a script for doing the RNN job

# ./py-sbatch.sh main.py run-nb --allow-errors Part1_Sequence.ipynb
./py-sbatch.sh main.py run-nb --allow-errors Part2_VAE.ipynb
# command:
# runipy ***notebook_name **out_name
# ./py-sbatch.sh main.py runipy Part2_VAE.ipynb Part2_out
