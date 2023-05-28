
import os
import torch
import argparse
from tqdm import tqdm
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
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

for level in ("neuron", "expert") :
    for concept in ("knowledge", "semantic", "task") :
        output_path = "Spearman/{}/{}/{}".format(args.model, level, concept)
        os.makedirs(output_path, exist_ok = True)

        for layer in tqdm(range(12) if args.model == "T5_step" or level == "neuron" else range(1, 12, 2)) :
            layer_ap = []
            for step in steps :
                if level == "neuron" :
                    if concept in ("knowledge", "semantic") :
                        ap = torch.load("../NeuronPredictivity/predictivity/{}/{}{}/{}.bin".format(concept, args.model, step, layer))
                    else :
                        ap = {}
                        for dataset in ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2") :
                            for index, index_ap in enumerate(torch.load("../NeuronPredictivity/predictivity/task/{}/{}{}/{}.bin".format(dataset, args.model, step, layer))) :
                                ap[dataset + str(index)] = index_ap
                else :
                    ap = torch.load("ExpertPredictivity/{}/{}{}/{}.bin".format(concept, args.model, step, layer))
                layer_ap.append(ap)
            mean_res = calc(layer_ap)
            torch.save(mean_res, os.path.join(output_path, "{}.bin".format(layer)))