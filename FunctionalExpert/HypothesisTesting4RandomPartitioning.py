
import argparse
import os
from tqdm import tqdm
import torch
from scipy import stats
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5", "switchT5"))
args = parser.parse_args()

def get_counting(ap, expert_belong) :
    res = torch.zeros((16, ))
    for neuron in ap :
        res[expert_belong[neuron]] += 1
    return res

expert_belong = []
for expert in range(16) :
    expert_belong += [expert] * (3072 if args.model == "switchT5" else 3072 // 16)

def binom_hypothesis_testing(countings : torch.Tensor) :
    rv = stats.binom(countings.sum(), 1.0 / 16)
    l, r = 0, countings.sum().item()
    while l < r :
        mid = (l + r) // 2
        if 1.0 - rv.cdf(mid) <= 0.001 :
            r = mid
        else :
            l = mid + 1
    threshold = l
    assert(1.0 - rv.cdf(threshold) <= 0.001)
    assert(1.0 - rv.cdf(threshold - 1) > 0.001)
    rejected = countings[countings > threshold]
    rejected_proportion = len(rejected) / 16
    modularization_extent = (rejected / (countings.sum() / 16)).mean().item() if len(rejected) else 0.0
    return rejected_proportion, modularization_extent

all_epoch = 200
for concept in ("knowledge", "semantic", "task") :
    os.makedirs("HypothesisTesting4RandomPartitioning/{}/{}".format(concept, args.model), exist_ok = True)
    for layer in tqdm((range(1, 12, 2) if args.model == "switchT5" else range(12))) :
        all_ap = torch.load("TopK_neurons/{}/{}/all_ap_{}.bin".format(concept, args.model, layer))
        sum_class = [0.0, 0.0]
        for epoch in range(all_epoch) :
            def Add(a, b) :
                a[0] += b[0] / all_epoch
                a[1] += b[1] / all_epoch
            random.shuffle(expert_belong)
            Add(sum_class, binom_hypothesis_testing(get_counting(all_ap, expert_belong)))
        torch.save(sum_class, "HypothesisTesting4RandomPartitioning/{}/{}/{}.bin".format(concept, args.model, layer))