import argparse
import os
from tqdm import tqdm
import torch
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5", "T5_step", "switchT5", "switchT5_step"))
parser.add_argument("--initialization", type = str, default = "pre-trained")
args = parser.parse_args()

def get_counting(ap, expert_belong) :
    res = torch.zeros((16, ))
    for neuron in ap :
        res[expert_belong[neuron]] += 1
    return res

if args.model in ("T5", "switchT5") :
    assert(args.initialization in ("pre-trained", "random"))
    if args.initialization == "random" :
        assert args.model == "T5"
    initialization = "" if args.initialization == "pre-trained" else "random"
elif args.model in ("T5_step", "switchT5_step") :
    initialization = args.initialization
else :
    raise NotImplementedError # impossible

if "switchT5" in args.model :
    expert_belong = []
    for expert in range(16) :
        expert_belong += [expert] * 3072
else :
    Expert_belong = []
    if args.model == "T5" :
        all_expert_belong = torch.load("../MoEfication/T5.bin")
        for layer in range(12) :
            Expert_belong.append(all_expert_belong["encoder.block.{}.layer.1.DenseReluDense.wi.weight".format(layer)])
    elif args.model == "T5_step" :
        for layer in range(12) :
            Expert_belong.append(torch.load("../MoEfication/T5_step/{}.bin".format(layer)))
    else :
        raise NotImplementedError

def hypergeom_hypothesis_testing(countings : torch.Tensor) :
    M = (3072 * 16 if "switchT5" in args.model else 3072)
    N = countings.sum()
    n = (3072 if "switchT5" in args.model else 3072 // 16)
    rv = stats.hypergeom(M, n, N)
    l, r = 0, n
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
    modularization_extent = (rejected / (N / 16)).mean().item() if len(rejected) else 0.0
    return rejected_proportion, modularization_extent

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

for concept in ("knowledge", "semantic", "task") :
    os.makedirs("HypothesisTesting/{}/{}".format(concept, args.model + initialization), exist_ok = True)
    for layer in (tqdm(range(1, 12, 2) if "switchT5" in args.model else range(12))) :
        ap = torch.load("TopK_neurons/{}/{}/ap_{}.bin".format(concept, args.model + initialization, layer))
        all_ap = torch.load("TopK_neurons/{}/{}/all_ap_{}.bin".format(concept, args.model + initialization, layer))
        if "switchT5" not in args.model :
            expert_belong = Expert_belong[layer]
        
        ap = {item : hypergeom_hypothesis_testing(get_counting(item_ap, expert_belong)) for item, item_ap in ap.items()}
        all_ap = binom_hypothesis_testing(get_counting(all_ap, expert_belong))

        torch.save(ap, "HypothesisTesting/{}/{}/sub-function_{}.bin".format(concept, args.model + initialization, layer))
        torch.save(all_ap, "HypothesisTesting/{}/{}/function_{}.bin".format(concept, args.model + initialization, layer))