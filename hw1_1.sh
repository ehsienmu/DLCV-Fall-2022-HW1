#!/bin/bash
wget https://www.dropbox.com/s/h7gbpzfdf9bd669/epoch_33.pt?dl=0 -O hw1_1_best_r10921a36.pt
# TODO - run your inference Python3 code
python3 p1_test.py --input_dir $1 --output_file $2 --model_file hw1_1_best_r10921a36.pt