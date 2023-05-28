import os
import sys
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

all_res = []
model_name = sys.argv[1] # ['switchT5', 'T5']
topk = 32
if 'switch' in model_name:
    topk *= 16

def sim_topk(a, b, k):
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    a_ind = np.argpartition(a, -k)[-k:]
    b_ind = np.argpartition(b, -k)[-k:]
    res = len(set(a_ind) & set(b_ind)) / k
    return res


os.makedirs("Figure1Right", exist_ok = True)

if 'switch' in model_name:
    layer_range = range(1, 12, 2)
else:
    layer_range = range(12)

for layer in layer_range:
    semantic_num = 0
    knowledge_num = 0
    task_num = 0

    plt.figure()

    semantic = torch.load("../NeuronPredictivity/predictivity/semantic/{0}/{1}.bin".format(model_name, layer))

    samples = []

    for k, v in semantic.items():
        samples.append(v)
        # if len(samples) == 30:
        #     break
    semantic_num = len(samples)

    knowledge = torch.load("../NeuronPredictivity/predictivity/knowledge/{0}/{1}.bin".format(model_name, layer))

    for k, v in knowledge.items():
        samples.append(v)
    
    knowledge_num = len(samples)


    # qqp
    qqp = torch.load("../NeuronPredictivity/predictivity/task/qqp/{0}/{1}.bin".format(model_name, layer))
    samples.append(qqp)

    # SST2
    sst2 = torch.load("../NeuronPredictivity/predictivity/task/sst2/{0}/{1}.bin".format(model_name, layer))
    samples.append(sst2)

    # mnli
    mnli = torch.load("../NeuronPredictivity/predictivity/task/mnli/{0}/{1}.bin".format(model_name, layer))
    samples.append(mnli)

    # cola
    cola = torch.load("../NeuronPredictivity/predictivity/task/cola/{0}/{1}.bin".format(model_name, layer))
    samples.append(cola)

    # mrpc
    mrpc = torch.load("../NeuronPredictivity/predictivity/task/mrpc/{0}/{1}.bin".format(model_name, layer))
    samples.append(mrpc)

    # rte
    rte = torch.load("../NeuronPredictivity/predictivity/task/rte/{0}/{1}.bin".format(model_name, layer))
    samples.append(rte)

    # qnli
    qnli = torch.load("../NeuronPredictivity/predictivity/task/qnli/{0}/{1}.bin".format(model_name, layer))
    samples.append(qnli)

    task_num = len(samples)

    new_samples = []
    for i in range(len(samples)):
        if len(samples[i]) == 1:
            new_samples.append(samples[i][0])
        elif len(samples[i]) == 2:
            new_samples.append(samples[i][0])
            new_samples.append(samples[i][1])
        else:
            new_samples.append(samples[i])

    samples = new_samples
    res = np.ndarray(shape=(len(samples), len(samples)))

    for i in range(len(samples)):
        for j in range(len(samples)):
            res[i][j] = sim_topk(samples[i], samples[j], topk)

    sns.heatmap(res)

    all_res.append(res)

plt.figure()

all_res = np.mean(np.stack(all_res), 0)

res = np.ndarray(shape=(3, 3))
refs = [0, semantic_num, knowledge_num, task_num]

for i in range(3):
    for j in range(3):
        if i != j:
            res[i][j] = np.mean(all_res[refs[i]:refs[i+1], refs[j]:refs[j+1]])
        else:
            tmp = np.sum(all_res[refs[i]:refs[i+1], refs[j]:refs[j+1]]) - (refs[i+1] - refs[i])
            res[i][j] = tmp / (refs[i+1] - refs[i] - 1) / (refs[j+1] - refs[j])

sns.heatmap(res, cmap="YlGnBu", annot=True, vmin=0.01, vmax=0.1)

plt.xticks([0.5, 1.5, 2.5], ["Semantic", "Knowledge", "Task"])
plt.yticks([0.5, 1.5, 2.5], ["Semantic", "Knowledge", "Task"])

plt.savefig("Figure1Right/{}.pdf".format("T5" if model_name == "T5" else "SwitchTransformer"))
plt.savefig("Figure1Right/{}.png".format("T5" if model_name == "T5" else "SwitchTransformer"))