#!/bin/bash

SCRIPT="$(pwd)/script/"
MODE="variance"
NSIMS=3

for i in $(seq 0 $((NSIMS-1)))
do
    echo $i
    # Runs simulation
    python3 ${SCRIPT}simulation/plasticity_simulation.py $MODE $i
done

# Processes data
python3 ${SCRIPT}processing/plasticity_analysis.py $MODE $NSIMS
python3 ${SCRIPT}processing/active_synapses.py $MODE $NSIMS

# Plots
python3 ${SCRIPT}visualization/plot_plasticity.py $MODE $NSIMS
python3 ${SCRIPT}visualization/plot_active_synapses.py $MODE $NSIMS
