
import os
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
args = parser.parse_args()

steps = list(range(0, 195000 + 1, 5000)) + [199999]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

os.makedirs("Figure4/{}".format("T5" if args.model == "T5_step" else "SwitchTransformer"), exist_ok = True)
for concept in ("knowledge", "semantic", "task") :
    for level in ("neuron", "expert", "random_partition") :
        all_mean_res = torch.zeros(size = (len(steps) - 1, ))
        for layer in tqdm(range(12) if args.model == "T5_step" or level == "neuron" else range(1, 12, 2)) :
            mean_res = torch.load("Spearman/{}/{}/{}/{}.bin".format(args.model, level, concept, layer))
            all_mean_res += mean_res / (12 if args.model == "T5_step" or level == "neuron" else 6)
        if level == "neuron" :
            neuron = all_mean_res
        elif level == "expert" :
            expert = all_mean_res
        elif level == "random_partition" :
            random_expert = all_mean_res
        else :
            raise NotImplementedError
    plt.figure()
    plt.xlabel("Step ($\\times 10^3$)")
    plt.ylabel("Stabilization Score")
    plt.tight_layout()
    plt.plot(torch.tensor(steps[: -1]) / 1000.0, expert, label = ("Post-MoE" if args.model == "T5_step" else "Pre-MoE"), linestyle = "solid", linewidth = 3)
    plt.plot(torch.tensor(steps[: -1]) / 1000.0, neuron, label = "Neuron", linestyle = "dotted", linewidth = 3)
    plt.plot(torch.tensor(steps[: -1]) / 1000.0, random_expert, label = "Random Partitioning", linestyle = "dashed", linewidth = 3)
    plt.legend(loc = "best")
    plt.savefig("Figure4/{}/{}.png".format("T5" if args.model == "T5_step" else "SwitchTransformer", concept), format = "png")
    plt.savefig("Figure4/{}/{}.pdf".format("T5" if args.model == "T5_step" else "SwitchTransformer", concept), format = "pdf")
    plt.close()