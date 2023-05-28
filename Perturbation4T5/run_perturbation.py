import os

template = "python glue_inf.py --gpu 0 --method {} --seed {} --percent {} --dataset {} --var {} --source {}"

var = 4
for seed in range(5):
    for percent in [0, 2, 6, 10, 12]:
        for task in ['multirc', 'cb', 'boolq', 'sst2', 'mrpc', 'cola', 'qqp', 'mnli', 'qnli', 'rte']:
            for method in ['ap']: 
                for source in ['sst2', 'mrpc', 'cola', 'qqp']:
                    os.system(template.format(method, seed, percent, task, var, source))
            for method in ['ap_avg', 'moe', 'random']:
                os.system(template.format(method, seed, percent, task, var, 'sst2'))
