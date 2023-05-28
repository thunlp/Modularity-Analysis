import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str)
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
parser.add_argument("--function", type = str, required = True)
args = parser.parse_args()

if args.gpu is None :
    command = "python run.py --model {} --function {}".format(args.model, args.function)
else :
    command = "python run.py --model {} --function {} --gpu {}".format(args.model, args.function, args.gpu)

for step in list(range(0, 200000, 5000)) + [199999] :
    os.system(command + " --initialization {}".format(step))