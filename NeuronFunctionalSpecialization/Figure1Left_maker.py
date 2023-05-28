import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

all_res = []
model_name = sys.argv[1] # ['switchT5', 'T5']
function_class = sys.argv[2] # ['semantic', 'knowledge', 'task']
folder = model_name

os.makedirs("Figure1Left".format(folder), exist_ok = True)

if 'switch' in model_name:
    layer_range = range(1, 12, 2)
else:
    layer_range = range(12)

for layer in layer_range:
    semantic_num = 0
    knowledge_num = 0
    task_num = 0

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

    res = torch.stack(new_samples)
    res = res.cpu().numpy().max(-1)
    res = [res[:semantic_num].mean(), res[semantic_num:knowledge_num].mean(), res[knowledge_num:].mean()]
    all_res.append(res)

if 'switch' in model_name:
    layer_range = range(2, 13, 2)
else:
    layer_range = range(1, 13)

model_name = model_name.split('_')[0]
all_res = np.array(all_res).T

if function_class == 'semantic':
    plt.plot(layer_range, all_res[0], label="Pre-trained", marker='o', markersize=10, linewidth=3, linestyle='-')
elif function_class == 'knowledge':
    plt.plot(layer_range, all_res[1], label="Pre-trained", marker='o', markersize=10, linewidth=3, linestyle='-')
elif function_class == 'task':
    plt.plot(layer_range, all_res[2], label="Pre-trained", marker='o', markersize=10, linewidth=3, linestyle='-')

all_res = []
model_name = model_name+'_step0'

if 'switch' in model_name:
    layer_range = range(1, 12, 2)
else:
    layer_range = range(12)

for layer in layer_range:
    semantic_num = 0
    knowledge_num = 0
    task_num = 0

    semantic = torch.load("../NeuronPredictivity/predictivity/semantic/{0}/{1}.bin".format(model_name, layer))

    samples = []

    for k, v in semantic.items():
        samples.append(v)
        if len(samples) == 30:
            break
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

    res = torch.stack(new_samples)
    res = res.cpu().numpy().max(-1)
    res = [res[:semantic_num].mean(), res[semantic_num:knowledge_num].mean(), res[knowledge_num:].mean()]
    all_res.append(res)

if 'switch' in model_name:
    layer_range = range(2, 13, 2)
else:
    layer_range = range(1, 13)

model_name = model_name.split('_')[0]
all_res = np.array(all_res).T

if function_class == 'semantic':
    plt.plot(layer_range, all_res[0], label="Random", marker='o', markersize=10, linewidth=3, linestyle='--')
elif function_class == 'knowledge':
    plt.plot(layer_range, all_res[1], label="Random", marker='o', markersize=10, linewidth=3, linestyle='--')
elif function_class == 'task':
    plt.plot(layer_range, all_res[2], label="Random", marker='o', markersize=10, linewidth=3, linestyle='--')

plt.legend()
plt.xlabel("Layer")
plt.ylabel("Best AP")

from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.xticks(range(2, 13, 2))


plt.savefig("Figure1Left/{}_{}.pdf".format("T5" if folder == "T5" else "SwitchTransformer", function_class), bbox_inches='tight')
plt.savefig("Figure1Left/{}_{}.png".format("T5" if folder == "T5" else "SwitchTransformer", function_class), bbox_inches='tight')