
import os
import torch
import random
import argparse
from tqdm import tqdm
from scipy import stats

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
parser.add_argument("--epoch", type = int, required = True)
args = parser.parse_args()

steps = list(range(0, 195000 + 1, 5000)) + [199999]

def calc(aps) :
    mean_res = torch.zeros(size = (len(steps) - 1, ))
    for item in aps[0] :
        for i, aps_i in enumerate(aps[: -1]) :
            j = i + 1
            aps_j = aps[j]
            mean_res[i] += stats.spearmanr(aps_i[item], aps_j[item])[0]
    return mean_res / len(aps[0])

for concept in ("knowledge", "semantic", "task") :
    output_path = "Spearman/{}/random_partition/{}".format(args.model, concept)
    os.makedirs(output_path, exist_ok = True)
    res = {layer : torch.zeros((len(steps) - 1, )) for layer in (range(12) if args.model == "T5_step" else range(1, 12, 2))}
    for epoch in tqdm(range(args.epoch)) :
        for layer in range(12) if args.model == "T5_step" else range(1, 12, 2) :
            expert_belong = []
            for expert in range(16) :
                expert_belong += [expert] * (3072 // 16 if args.model == "T5_step" else 3072)
            random.shuffle(expert_belong)
            expert_belong = torch.LongTensor(expert_belong)
            all_ap = []
            for step in steps :
                if concept in ("knowledge", "semantic") :
                    ap = torch.load("../NeuronPredictivity/predictivity/{}/{}{}/{}.bin".format(concept, args.model, step, layer))
                else :
                    ap = {}
                    for dataset in ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2") :
                        for index, index_ap in enumerate(torch.load("../NeuronPredictivity/predictivity/task/{}/{}{}/{}.bin".format(dataset, args.model, step, layer))) :
                            ap[dataset + str(index)] = index_ap
                ap = {item : [item_ap[expert_belong == expert].mean() for expert in range(16)] for item, item_ap in ap.items()}
                all_ap.append(ap)
            res[layer] += calc(all_ap) / args.epoch
    all_res = torch.zeros((len(steps) - 1, ))
    for layer in range(12) if args.model == "T5_step" else range(1, 12, 2) :
        torch.save(res[layer], os.path.join(output_path, "{}.bin".format(layer)))
        all_res += res[layer] / (12 if args.model == "T5_step" else 6)