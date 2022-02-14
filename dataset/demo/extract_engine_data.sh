#!/bin/bash
# Extract data of a specific engine id from dataset
# The command below extract data of engine id = 20

dataset=test_FD001
# dataset=test_FD004

echo "extracting data from ../CMAPSSData/${dataset}.txt"
awk '/^20[ \t]+/ {print}' ../CMAPSSData/${dataset}.txt > ${dataset}_e020.txt
