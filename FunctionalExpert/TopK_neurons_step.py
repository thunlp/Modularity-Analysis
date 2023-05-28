import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
args = parser.parse_args()

command = "python TopK_neurons.py --model {}".format(args.model)

for step in list(range(0, 200000, 5000)) + [199999] :
    os.system(command + " --initialization {}".format(step))