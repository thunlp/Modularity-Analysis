
import os
import types
import torch
import random
import argparse
import datasets
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

def forward_with_noise(var=2):

    def _forward(model_self, hidden_states):
        hidden_states = model_self.wi(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        noise = hidden_states.detach().clone()[:, :, model_self.mask == True]
        bsz, seq_len, hidden_size = noise.shape
        noise = noise.view(-1)
        noise = noise[torch.randperm(hidden_size*bsz*seq_len)]
        noise = noise.view(bsz, seq_len, hidden_size)
        hidden_states[:, :, model_self.mask == True] = noise * var

        hidden_states = torch.nn.functional.relu(hidden_states)

        hidden_states = model_self.dropout(hidden_states)
        hidden_states = model_self.wo(hidden_states)
        return hidden_states

    return _forward

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, default = None)
parser.add_argument('--method', type=str, default='ap')
parser.add_argument('--source', type=str, default='')
parser.add_argument('--percent', type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='sst2')
parser.add_argument('--var', type=float, default=1)

args = parser.parse_args()

def set_gpu(gpu : str) :
    gpu_list = []
    if gpu is None :
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else :
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        device_list = gpu.split(",")
        for number in range(len(device_list)) :
            gpu_list.append(int(number))
    
    cuda = torch.cuda.is_available()
    if not cuda and len(gpu_list) :
        raise NotImplementedError
set_gpu(args.gpu)

os.environ["PL_GLOBAL_SEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() :
    torch.cuda.manual_seed_all(args.seed)

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')
if torch.cuda.is_available() :
    model = model.cuda()

res = torch.load('t5_encoder_cluster_96_weight.bin')
task_expert = torch.load('t5_96_task_expert.bin')

mask_nums = []

for layer in range(8, 12):
    for k, v in model.named_modules():
        if k == "encoder.block.{}.layer.1.DenseReluDense".format(layer):

            tmp_res = res[k+'.wi.weight']

            if not args.source:
                args.source = args.dataset

            dat_ap = torch.load("../NeuronPredictivity/predictivity/task/{}/T5/{}.bin".format(args.source, layer))

            if args.method == 'moe':
                sums = task_expert[layer]
                threshold = np.percentile(sums, (1-(args.percent*32/3072))*100)

                mask = [False]*len(tmp_res)
                for i in range(len(tmp_res)):
                    if sums[tmp_res[i]] > threshold:
                        mask[i] = True
                v.mask = torch.tensor(mask)

            if args.method == 'ap':
                threshold = np.percentile(dat_ap[0].numpy(), (1-(args.percent*32/3072))*100)
                v.mask = dat_ap[0] > threshold

            if args.method == 'ap_avg':
                dat_ap = torch.load("average_predictivity/{}.bin".format(layer))
                threshold = np.percentile(dat_ap, (1-(args.percent*32/3072))*100)
                v.mask = dat_ap > threshold

            if args.method == 'random':
                threshold = np.percentile(dat_ap[0].numpy(), (1-(args.percent*32/3072))*100)
                v.mask = dat_ap[0] > threshold
                idx = torch.randperm(v.mask.nelement())
                v.mask = v.mask.view(-1)[idx].view(v.mask.size())

            mask_nums.append(v.mask.sum())

            v._forward = v.forward
            v.forward = types.MethodType(forward_with_noise(args.var), v)

if args.dataset == "mnli" :
    file = datasets.load_dataset("glue", args.dataset)["validation_matched"]
elif args.dataset in ('sst2', 'mrpc', 'cola', 'qqp', 'rte', 'qnli') :
    file = datasets.load_dataset("glue", args.dataset)["validation"]
elif args.dataset in ('cb', 'copa', 'boolq', 'multirc') :
    file = datasets.load_dataset("super_glue", args.dataset)["validation"]
else:
    raise NotImplementedError

sentences = []
labels = []

for instance in file :
    if args.dataset == 'sst2':
        sentences.append(instance['sentence'])
        labels.append(instance['label'])
    elif args.dataset == 'mrpc':
        sentences.append([instance['sentence1'], instance['sentence2']])
        labels.append(instance['label'])
    elif args.dataset == 'cola':
        sentences.append(instance['sentence'])
        labels.append(instance['label'])
    elif args.dataset == 'qqp':
        sentences.append([instance['question1'], instance['question2']])
        labels.append(instance['label'])
    elif args.dataset == 'rte':
        sentences.append([instance['sentence1'], instance['sentence2']])
        labels.append(1 - instance['label'])
    elif args.dataset == 'mnli':
        sentences.append([instance['premise'], instance['hypothesis']])
        labels.append(2 - instance['label'])
    elif args.dataset == 'qnli':
        sentences.append([instance['question'], instance['sentence']])
        labels.append(1 - instance['label'])
    elif args.dataset == 'cb':
        sentences.append([instance['hypothesis'], instance['premise']])
        labels.append(instance['label'])
    elif args.dataset == 'copa':
        sentences.append([instance['choice1'], instance['choice2'], instance['premise'], instance['question']])
        labels.append(instance['label'])
    elif args.dataset == 'boolq':
        sentences.append([instance['question'], instance['passage']])
        labels.append(instance['label'])
    elif args.dataset == 'multirc':
        sentences.append([instance['question'], instance['paragraph'], instance['answer']])
        labels.append(instance['label'])
    else:
        raise NotImplementedError


pred = []
model.eval()

batch_size = 4

for i in tqdm(range(0, len(sentences), batch_size)):

    sents = sentences[i:i+batch_size]
    inputs = []
    for sent in sents:
        if args.dataset == 'sst2':
            inputs.append("sst2 sentence: {}".format(sent))
        elif args.dataset == 'mrpc':
            inputs.append("mrpc sentence1: {} sentence2: {}".format(sent[0], sent[1]))
        elif args.dataset == 'cola':
            inputs.append("cola sentence: {}".format(sent))
        elif args.dataset == 'qqp':
            inputs.append("qqp question1: {} question2: {}".format(sent[0], sent[1]))
        elif args.dataset == 'rte':
            inputs.append("rte sentence1: {} sentence2: {}".format(sent[0], sent[1]))
        elif args.dataset == 'mnli':
            inputs.append("mnli hypothesis: {} premise: {}".format(sent[1], sent[0]))
        elif args.dataset == 'qnli':
            inputs.append("qnli question: {} sentence: {}".format(sent[0], sent[1]))
        elif args.dataset == 'cb':
            inputs.append("cb hypothesis: {} premise: {}".format(sent[0], sent[1]))
        elif args.dataset == 'copa':
            inputs.append("copa choice1: {} choice2: {} premise: {} question: {}".format(sent[0], sent[1], sent[2], sent[3]))
        elif args.dataset == 'boolq':
            inputs.append("boolq question: {} passage: {}".format(sent[0], sent[1]))
        elif args.dataset == 'multirc':
            inputs.append("multirc question: {} answer: {} paragraph: {} ".format(sent[0], sent[2], sent[1]))
        

    encoding = tokenizer(inputs, return_tensors="pt", padding=True)

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    dec_input_ids = tokenizer(["<extra_id_0>"]*len(inputs), return_tensors="pt").input_ids[:, :1]
    if torch.cuda.is_available():
        input_ids, attention_mask, dec_input_ids = input_ids.cuda(), attention_mask.cuda(), dec_input_ids.cuda()

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=dec_input_ids)

    for i in range(len(inputs)):
        if args.dataset == 'sst2':
            cur_pred = [output.logits[i, 0, 1465].item(), output.logits[i, 0, 2841].item()]
        elif args.dataset == 'mrpc':
            cur_pred = [output.logits[i, 0, 7072].item(), output.logits[i, 0, 59].item()]
        elif args.dataset == 'cola':
            cur_pred = [output.logits[i, 0, 9961], output.logits[i, 0, 29452]]
        elif args.dataset == 'qqp':
            cur_pred = [output.logits[i, 0, 19197].item(), output.logits[i, 0, 59].item()]
        elif args.dataset == 'rte':
            cur_pred = [output.logits[i, 0, 3].item(), output.logits[i, 0, 59].item()]
        elif args.dataset == 'mnli':
            cur_pred = [output.logits[i, 0, 3].item(), output.logits[i, 0, 7163].item(), output.logits[i, 0, 27252].item()]
        elif args.dataset == 'qnli':
            cur_pred = [output.logits[i, 0, 3].item(), output.logits[i, 0, 59].item()]
        elif args.dataset == 'cb':
            cur_pred = [output.logits[i, 0, 3].item(), output.logits[i, 0, 27252].item(), output.logits[i, 0, 7163].item()]
        elif args.dataset == 'copa':
            cur_pred = [output.logits[i, 0, 10998].item(), output.logits[i, 0, 10747].item()]
        elif args.dataset == 'boolq':
            cur_pred = [output.logits[i, 0, 10998].item(), output.logits[i, 0, 10747].item()]
        elif args.dataset == 'multirc':
            cur_pred = [output.logits[i, 0, 10998].item(), output.logits[i, 0, 10747].item()]

        if cur_pred[0] > cur_pred[1]:
            pred.append(1)
        else:
            pred.append(0)
        
        if args.dataset == 'mnli':
            cur_pred = 2-np.argmax(cur_pred)
            pred[-1] = cur_pred
        elif args.dataset == 'cb':
            cur_pred = np.argmax(cur_pred)
            pred[-1] = cur_pred

acc = sum([1 for x, y in zip(pred, labels) if int(x) == int(y)]) / len(labels)
print(args.dataset, args.method, args.percent, args.seed, args.var, args.source, acc, mask_nums)
with open('results.txt', 'a') as f:
    print(args.dataset, args.method, args.percent, args.seed, args.var, args.source, acc, file=f)
