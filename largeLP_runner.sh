#!/bin/bash
mkdir -p large-problem-instances

python generate_multicommodity_flow.py \
    --output_file large-problem-instances/multicommodity-flow-instance.mps \
    --num_commodities 1000 \
    --num_warehouses 100 \
    --num_stores 1000 \
    --seed 1

python generate_heat_source_location.py \
    --output_file large-problem-instances/heat-source-instance-easy.mps \
    --ground_truth_file large-problem-instances/temperature_ground_truth-easy.h5 \
    --grid_size 100 \
    --num_source_locations 5 \
    --num_possible_source_locations 300 \
    --num_measurement_locations 700 \
    --seed 1

# Add other calls similarly...