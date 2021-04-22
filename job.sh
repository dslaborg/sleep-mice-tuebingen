#!/bin/sh
python scripts/transform_files_tuebingen.py -e exp001
python scripts/evaluate_experiment.py -e exp001 -d valid
