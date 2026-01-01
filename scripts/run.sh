#!/bin/bash

uv run src/png2coordinate.py
uv run src/noise_rotate.py

cp data/secret_akeome_10d.csv 10d_data
