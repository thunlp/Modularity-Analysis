import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from NeuronProcessor import NeuronProcessor

def Observe(sentences, model, tokenizer, neuron_processor : NeuronProcessor, observe_function, res) :
    model.eval()
    batch_size = 8
    for index in tqdm(range(len(sentences) // batch_size + 1)) :
        batch_sentences = sentences[index * batch_size : (index + 1) * batch_size]
        if len(batch_sentences) == 0 :
            continue
        encoded_input = tokenizer(batch_sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        if torch.cuda.is_available() :
            for k, v in encoded_input.items() :
                encoded_input[k] = v.cuda()
        with torch.no_grad() :
            hooks = observe_function(model, res, encoded_input["attention_mask"])
            model(**encoded_input)
            neuron_processor.erase_hooks(hooks)
    return res

def ap_Observe(sentences, model, tokenizer, neuron_processor : NeuronProcessor) :
    res = defaultdict(list)
    res = Observe(sentences, model, tokenizer, neuron_processor, neuron_processor.observe, res)
    return res

def concept_Observe(data_path, model, tokenizer, neuron_processor : NeuronProcessor, Filter = lambda key : True) :
    sentences = []
    with open(data_path) as fin :
        for line in fin.readlines() :
            line = line.strip().split("\t")
            if Filter(line[0]) :
                sentences.append(line[-1])
    return ap_Observe(sentences, model, tokenizer, neuron_processor)

def task_Observe(task, model, tokenizer, neuron_processor : NeuronProcessor, separator) :
    TRAIN = False
    if task.startswith("TRAIN") :
        TRAIN = True
        task = task[len("TRAIN") :]
    def task_format(task, instance) :
        if task == "cola" :
            return "cola sentence: {}".format(instance["sentence"])
        elif task == "rte" :
            return "rte sentence1: {}{}sentence2: {}".format(instance["sentence1"], separator, instance["sentence2"])
        elif task == "mnli" :
            return "mnli hypothesis: {}{}premise: {}".format(instance["premise"], separator, instance["hypothesis"])
        elif task == "mrpc" :
            return "mrpc sentence1: {}{}sentence2: {}".format(instance["sentence1"], separator, instance["sentence2"])
        elif task == "qnli" :
            return "qnli question: {}{}sentence: {}".format(instance["question"], separator, instance["sentence"])
        elif task == "qqp" :
            return "qqp question1: {}{}question2: {}".format(instance["question1"], separator, instance["question2"])
        elif task == "sst2" : 
            return "sst2 sentence: {}".format(instance["sentence"])
        else :
            assert(False)
    with open("data/task/{}/{}.json".format(task, "train" if TRAIN else "validation")) as fin :
        dataset = json.load(fin)
    sentences = []
    for label, subset in dataset.items() :
        if len(subset) > 1000 :
            subset = random.sample(subset, 1000)
            dataset[label] = subset
        for instance in subset :
            instance["idx"] = len(sentences)
            sentences.append(task_format(task, instance))
    return ap_Observe(sentences, model, tokenizer, neuron_processor), dataset