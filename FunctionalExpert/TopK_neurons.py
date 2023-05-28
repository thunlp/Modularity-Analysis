
import os
import torch
import argparse

def get_TopK(ap : torch.Tensor) :
    return [x.item() for x in ap.topk(int(len(ap) * 0.01)).indices]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True, choices = ("T5", "T5_step", "switchT5", "switchT5_step"))
parser.add_argument("--initialization", type = str, default = "pre-trained")
args = parser.parse_args()

if args.model in ("T5", "switchT5") :
    assert(args.initialization in ("pre-trained", "random"))
    if args.initialization == "random" :
        assert args.model == "T5"
    initialization = "" if args.initialization == "pre-trained" else "random"
elif args.model in ("T5_step", "switchT5_step") :
    initialization = args.initialization
else :
    raise NotImplementedError # impossible

for concept in ("knowledge", "semantic", "task") :
    os.makedirs("TopK_neurons/{}/{}".format(concept, args.model + initialization), exist_ok = True)
    for layer in (range(1, 12, 2) if "switchT5" in args.model else range(12)) :
        if concept in ("knowledge", "semantic") :
            ap = torch.load("../NeuronPredictivity/predictivity/{}/{}/{}.bin".format(concept, args.model + initialization, layer))
        else :
            ap = {}
            for dataset in ("cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2") :
                for index, index_ap in enumerate(torch.load("../NeuronPredictivity/predictivity/task/{}/{}/{}.bin".format(dataset, args.model + initialization, layer))) :
                    ap[dataset + str(index)] = index_ap
        ap = {item : get_TopK(item_ap) for item, item_ap in ap.items()}
        torch.save(ap, "TopK_neurons/{}/{}/ap_{}.bin".format(concept, args.model + initialization, layer))
        all_ap = []
        for item_ap in ap.values() :
            all_ap += item_ap
        torch.save(all_ap, "TopK_neurons/{}/{}/all_ap_{}.bin".format(concept, args.model + initialization, layer))