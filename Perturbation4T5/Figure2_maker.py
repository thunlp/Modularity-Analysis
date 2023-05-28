import os
import matplotlib.pyplot as plt

os.makedirs("Figure2", exist_ok = True)

scale = 4

from collections import defaultdict
d = defaultdict(list)
with open('results.txt') as fin:
    for line in fin:
        line = line.strip().split()
        name = " ".join([line[0], line[1], line[2], line[-2]])
        if float(line[-3]) != scale:
            continue
        d[name].append(line[-1])

for k in d:
    d[k] = sum([float(i) for i in d[k]]) / len(d[k])

def plot_func(constraints, filename):
    targets = ['sst2', 'mrpc', 'cola', 'qqp']
    numbers = ['0', '2', '6', '10', '12']
    d_res = {}
    for target in targets:
        print(target)
        tmp_d = defaultdict(list)
        for k, v in d.items():
            vec = k.split()
            if vec[-1] == target and vec[1] == 'ap':
                if constraints and not vec[0] in constraints:
                    continue
                tmp_d[vec[2]].append(v)
        for k in tmp_d:
            tmp_d[k] = sum([float(i) for i in tmp_d[k]]) / len(tmp_d[k])
        d_res[target] = [x[1] for x in sorted(tmp_d.items(), key=lambda x: int(x[0])) if x[0] in numbers]

    targets = ['moe', 'random', 'ap_avg']
    for target in targets:
        print(target)
        tmp_d = defaultdict(list)
        for k, v in d.items():
            vec = k.split()
            if vec[1] == target:
                if constraints and not vec[0] in constraints:
                    continue
                tmp_d[vec[2]].append(v)
        for k in tmp_d:
            tmp_d[k] = sum([float(i) for i in tmp_d[k]]) / len(tmp_d[k])
        d_res[target] = [x[1] for x in sorted(tmp_d.items(), key=lambda x: int(x[0])) if x[0] in numbers]

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    for target in ['random', 'sst2', 'mrpc', 'cola', 'qqp', 'ap_avg', 'moe']:
        tmp_res = d_res[target]
        if target == 'ap_avg':
            target = 'avg' 
        if target != 'random':
            tmp_res = d_res['random'][:1]+tmp_res
        
        if target == 'sst2':
            target = 'SST-2'
        elif target == 'mrpc':
            target = 'MRPC'
        elif target == 'cola':
            target = 'CoLA'
        elif target == 'qqp':
            target = 'QQP'
        elif target == 'moe':
            target = 'MoE'
        elif target == 'random':
            target = 'Random'
        elif target == 'avg':
            target = 'Avg'

        plt.plot(tmp_res, label=target, marker='o', linewidth=3, markersize=10)
    plt.legend()

    plt.xticks(range(0, 5), [ str(x) for x in [0, 2, 6, 10, 12]])
    plt.xlabel('Perturbation Percentage')
    plt.ylabel('Averaged Accuracy')

    plt.savefig(filename + ".png")
    plt.savefig(filename + ".pdf")


plot_func(['sst2', 'mrpc', 'cola', 'qqp'], 'Figure2/fig2a')
plot_func(['mnli', 'qnli', 'cb', 'multirc', 'boolq'], 'Figure2/fig2b')