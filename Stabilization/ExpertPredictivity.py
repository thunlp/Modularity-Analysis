
import os
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5_step", "switchT5_step"))
args = parser.parse_args()

if args.model == "switchT5_step" :
    expert_belong = []
    for expert in range(16) :
        expert_belong += [expert] * 3072
    expert_belong = torch.LongTensor(expert_belong)
elif args.model == "T5_step" :
    Expert_belong = []
    for layer in range(12) :
        Expert_belong.append(torch.load("../MoEfication/T5_step/{}.bin".format(layer)))
else :
    raise NotImplementedError # impossible

for concept in ("knowledge", "semantic", "task") :
    for step in tqdm(list(range(0, 195000 + 1, 5000)) + [199999]) :
        os.makedirs("ExpertPredictivity/{}/{}{}".format(concept, args.model, step), exist_ok = True)
        all_Mean = []
        for layer in range(12) if args.model == "T5_step" else range(1, 12, 2) :
            if args.model == "T5_step" :
                expert_belong = Expert_belong[layer]
            if concept in ("knowledge", "semantic") :
                ap = torch.load("../NeuronPredictivity/predictivity/{}/{}{}/{}.bin".format(concept, args.model, step, layer))
            else :
                ap = {}
                for dataset in ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2") :
                    for index, index_ap in enumerate(torch.load("../NeuronPredictivity/predictivity/task/{}/{}{}/{}.bin".format(dataset, args.model, step, layer))) :
                        ap[dataset + str(index)] = index_ap
            Mean = {}
            for item, item_ap in ap.items() :
                Mean[item] = [], []
                expert_num = 16
                Mean[item] = torch.tensor([item_ap[expert_belong == expert].mean() for expert in range(expert_num)])
            all_Mean.append(Mean)
            torch.save(Mean, "ExpertPredictivity/{}/{}{}/{}.bin".format(concept, args.model, step, layer))