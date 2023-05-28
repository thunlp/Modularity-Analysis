import torch
import statistics

for strategy in ("Original", "NoFunction", "Function") :
    print("strategy = {}".format(strategy))
    task_avg = 0.0
    for dataset in ("cola", "mnli", "mrpc", "rte", "qnli", "qqp", "sst2") :
        Mean = sum([torch.tensor(torch.load("save/{}_128training/{}/validation_{}.bin".format(dataset, strategy, seed))) for seed in range(8)]) / 8 * 100
        print("{} : {}~($\pm${})".format(dataset, round(Mean[-1].item(), 2), round(statistics.stdev([torch.tensor(torch.load("save/{}_128training/{}/validation_{}.bin".format(dataset, strategy, seed)))[-1].item() for seed in range(8)]), 2)))
        task_avg += Mean[-1].item() / 7.0
    print("task_avg : {}".format(round(task_avg, 2)))
    print("\n")