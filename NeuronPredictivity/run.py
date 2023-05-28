import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from switchT5 import get_switchT5
from NeuronProcessor import NeuronProcessor
from observe import concept_Observe, task_Observe
from ap_calculation import task_CalAP, knowledge_CalAP, semantic_CalAP

import logging
logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt = "%m/%d/%Y %H:%M:%S",
                    level = logging.INFO)
logger = logging.getLogger(__name__)
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
    logger.info("CUDA available: {}".format(str(cuda)))
    if not cuda and len(gpu_list) :
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

def set_seed(seed : int) :
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(seed)

class T5_step_Tokenizer() :
    def __init__(self, tokenizer) :
        self.tokenizer = tokenizer
    def __call__(self, batch_sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512) :
        encoded_input = self.tokenizer(batch_sentences, return_tensors = return_tensors, padding = padding, truncation = truncation, max_length = max_length)
        for input_ids, attention_mask in zip(encoded_input["input_ids"], encoded_input["attention_mask"]) :
            for i in range(len(input_ids) - 1, -1, -1) :
                if input_ids[i] == 1 :
                    input_ids[i] = attention_mask[i] = 0
                    break
        return encoded_input

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = str)
    parser.add_argument("--model", type = str, required = True, choices = ("T5", "T5_step", "switchT5", "switchT5_step"))
    parser.add_argument("--function", type = str, required = True)
    parser.add_argument("--initialization", type = str, default = "pre-trained")
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()

    set_gpu(args.gpu)
    set_seed(args.seed)

    if args.model in ("T5", "switchT5") :
        assert(args.initialization in ("pre-trained", "random"))
        if args.initialization == "random" :
            assert args.model == "T5"
        initialization = "" if args.initialization == "pre-trained" else "random"
    elif args.model in ("T5_step", "switchT5_step") :
        initialization = args.initialization
    else :
        raise NotImplementedError # impossible
    
    if args.model == "T5" :
        path = "t5-base"
        tokenizer = T5Tokenizer.from_pretrained(path)
        if args.initialization == "pre-trained" :
            model = T5EncoderModel.from_pretrained(path)
        else :
            model = T5EncoderModel(T5Config.from_pretrained(path))
        if torch.cuda.is_available() :
            model = model.cuda()
    elif args.model == "switchT5" :
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = get_switchT5(device = "cuda" if torch.cuda.is_available() else "cpu", checkpoint = "checkpoint/switchT5/transformed_ckpt.bin")
    elif args.model == "T5_step" :
        tokenizer = T5_step_Tokenizer(T5Tokenizer.from_pretrained("t5-base"))
        model = T5EncoderModel(T5Config.from_pretrained("t5-base"))
        names = [key for key, value in model.named_parameters()]
        names.append("encoder.embed_tokens.weight")
        matched_params = torch.load("checkpoint/T5_step/{}/transformed_ckpt.bin".format(initialization), map_location = torch.device("cpu"))
        model.load_state_dict({k : torch.tensor(matched_params[k]) for k in matched_params if k in names})
        if torch.cuda.is_available() :
            model = model.cuda()
    elif args.model == "switchT5_step" :
        tokenizer = T5_step_Tokenizer(T5Tokenizer.from_pretrained("t5-base"))
        model = get_switchT5(device = "cuda" if torch.cuda.is_available() else "cpu", checkpoint = "checkpoint/switchT5_step/{}/transformed_ckpt.bin".format(initialization), tie_word_embeddings = False)
    else :
        raise NotImplementedError # impossible
    neuron_processor = NeuronProcessor()

    if args.function == "semantic" :
        semantic_data = "data/semantic/extracted_samples.txt"
        semantic_words = set()
        with open(semantic_data) as fin :
            for line in fin.readlines() :
                line = line.strip().split("\t")
                semantic_words.add(line[0].split(".")[0])
        if args.model == "switchT5_step" :
            with open("data/semantic/semantic4switchT5_step.json", "r") as fin :
                semantic_words = json.load(fin)
        os.makedirs("predictivity/semantic/{}".format(args.model + initialization), exist_ok = True)
        for cur_word in tqdm(semantic_words) :
            print("cur_word : {}".format(cur_word))
            result = concept_Observe(semantic_data, model, tokenizer, neuron_processor, Filter = lambda word : word.split(".")[0] == cur_word)
            for observe_layer in range(12) :
                print("observe_layer = {}".format(observe_layer))
                try :
                    record = torch.load("predictivity/semantic/{}/{}.bin".format(args.model + initialization, observe_layer))
                except :
                    record = {}
                if cur_word in record :
                    continue
                temp_record = semantic_CalAP(torch.cat(result[observe_layer], dim = 0), Filter = lambda word : word == cur_word)
                if cur_word in temp_record :
                    record[cur_word] = temp_record[cur_word]
                torch.save(record, "predictivity/semantic/{}/{}.bin".format(args.model + initialization, observe_layer))
        exit(0)
    
    if args.function == "knowledge" :
        knowledge_data = "data/knowledge/extracted_triples.txt"
        with open("data/knowledge/knowledge.json", "r") as fin :
            knowledge_relation = json.load(fin)
        os.makedirs("predictivity/knowledge/{}".format(args.model + initialization), exist_ok = True)
        for cur_relation in tqdm(knowledge_relation) :
            print("cur_relation : {}".format(cur_relation))
            result = concept_Observe(knowledge_data, model, tokenizer, neuron_processor, Filter = lambda relation : relation == cur_relation)
            for observe_layer in range(12) :
                print("observe_layer = {}".format(observe_layer))
                try :
                    record = torch.load("predictivity/knowledge/{}/{}.bin".format(args.model + initialization, observe_layer))
                except :
                    record = {}
                if cur_relation in record :
                    continue
                temp_record = knowledge_CalAP(torch.cat(result[observe_layer], dim = 0), Filter = lambda relation : relation == cur_relation)
                if cur_relation in temp_record :
                    record[cur_relation] = temp_record[cur_relation]
                torch.save(record, "predictivity/knowledge/{}/{}.bin".format(args.model + initialization, observe_layer))
        exit(0)
    
    if args.function.startswith("TASK") :
        result, dataset = task_Observe(args.function[len("TASK") :], model, tokenizer, neuron_processor, separator = " ")
        os.makedirs("predictivity/task/{}/{}".format(args.function[len("TASK") :], args.model + initialization), exist_ok = True)
    else :
        raise NotImplementedError
    for observe_layer in range(12) :
        record = {}
        if not os.path.exists("predictivity/task/{}/{}/{}.bin".format(args.function[len("TASK") :], args.model + initialization, observe_layer)) :
            record = task_CalAP(dataset, args.function[len("TASK") :], torch.cat(result[observe_layer], dim = 0))
            torch.save(record, "predictivity/task/{}/{}/{}.bin".format(args.function[len("TASK") :], args.model + initialization, observe_layer))
        result[observe_layer] = None