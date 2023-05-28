import os
import torch
import numpy as np

os.makedirs("average_predictivity", exist_ok = True)

for layer in range(12):
    t_X_new = []

    # qqp
    qqp = torch.load("../NeuronPredictivity/predictivity/task/qqp/T5/{}.bin".format(layer))
    t_X_new.append(qqp)

    # SST2
    sst2 = torch.load("../NeuronPredictivity/predictivity/task/sst2/T5/{}.bin".format(layer))
    t_X_new.append(sst2)

    # mnli
    mnli = torch.load("../NeuronPredictivity/predictivity/task/mnli/T5/{}.bin".format(layer))
    t_X_new.append(mnli)

    # cola
    cola = torch.load("../NeuronPredictivity/predictivity/task/cola/T5/{}.bin".format(layer))
    t_X_new.append(cola)

    # mrpc
    mrpc = torch.load("../NeuronPredictivity/predictivity/task/mrpc/T5/{}.bin".format(layer))
    t_X_new.append(mrpc)

    # rte
    rte = torch.load("../NeuronPredictivity/predictivity/task/rte/T5/{}.bin".format(layer))
    t_X_new.append(rte)

    # qnli
    qnli = torch.load("../NeuronPredictivity/predictivity/task/qnli/T5/{}.bin".format(layer))
    t_X_new.append(qnli)

    X = []
    for aps in t_X_new:
        for ap in aps:
            X.append(ap)

    X = np.stack(X)
    X = X.transpose()

    X = X.sum(axis=-1)
    torch.save(X, "average_predictivity/{}.bin".format(layer))