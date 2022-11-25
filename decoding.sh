#!/bin/bash

SCRIPT="$(pwd)/script/"
NSIMS=3

# Processes data
python3 ${SCRIPT}simulation/decoding_plasticity.py $NSIMS
python3 ${SCRIPT}simulation/decoding_ideal.py $NSIMS

# Plots
python3 ${SCRIPT}visualization/plot_decoding.py
