#!/bin/bash
wget https://www.dropbox.com/s/33ijfsbgy2qp6kd/part2_best.pt?dl=0 -O hw1_2_best_r10921a36.pt
# TODO - run your inference Python3 code
python3 p2_test.py --input_dir $1 --output_dir $2 --model_file hw1_2_best_r10921a36.pt