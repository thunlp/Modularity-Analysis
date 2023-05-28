
import os
import torch
import random
import datasets
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
from transformers import SwitchTransformersForConditionalGeneration, SwitchTransformersConfig, T5Tokenizer, SwitchTransformersTop1Router

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

def seed_everything(seed : int) :
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(seed)

def build_backbone() :
    path = "google/switch-base-16"
    tokenizer = T5Tokenizer.from_pretrained(path)
    config = SwitchTransformersConfig.from_pretrained(path)
    config.dropout_rate = 0.1
    model = SwitchTransformersForConditionalGeneration.from_pretrained(path, config = config)
    if torch.cuda.is_available() :
        model = model.cuda()
    return model, tokenizer

def load_dataset(args, split) :
    assert split in ("training", "validation")
    if args.dataset == "mnli" :
        file = datasets.load_dataset("glue", args.dataset)["train" if split == "training" else "validation_matched"]
    else :
        file = datasets.load_dataset("glue", args.dataset)["train" if split == "training" else "validation"]
    dataset = []
    for instance in file :
        if args.dataset == "sst2" :
            dataset.append({"sentence" : "{} It was <extra_id_0> .".format(instance["sentence"]), "label" : instance["label"]})
        elif args.dataset == "mrpc" :
            dataset.append({"sentence" : "{} <extra_id_0> , {}".format(instance["sentence1"], instance["sentence2"]), "label" : instance["label"]})
        elif args.dataset == "cola" :
            dataset.append({"sentence" : "{} This is <extra_id_0> .".format(instance["sentence"]), "label" : instance["label"]})
        elif args.dataset == "qqp" :
            dataset.append({"sentence" : "{} <extra_id_0> , {}".format(instance["question1"], instance["question2"]), "label" : instance["label"]})
        elif args.dataset == "rte" :
            dataset.append({"sentence" : "{} ? <extra_id_0> , {}".format(instance["sentence1"], instance["sentence2"]), "label" : 1 - instance["label"]})
        elif args.dataset == "mnli" :
            dataset.append({"sentence" : "{} ? <extra_id_0> , {}".format(instance["hypothesis"], instance["premise"]), "label" : 2 - instance["label"]})
        elif args.dataset == 'qnli' :
            dataset.append({"sentence" : "{} ? <extra_id_0> , {}".format(instance["question"], instance["sentence"]), "label" : 1 - instance["label"]})
        else :
            raise NotImplementedError

    return dataset

class biased_Top1Router(SwitchTransformersTop1Router) :
    def __init__(self, config: SwitchTransformersConfig, selected_experts: torch.LongTensor) :
        super().__init__(config)
        self.selected_experts = selected_experts
    def forward(self, hidden_states: torch.Tensor) :
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        self.selected_experts = self.selected_experts.to(router_probs.device)
        expert_index = self.selected_experts[torch.argmax(router_probs[:, :, self.selected_experts], dim = -1)] # expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        # token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        # expert_capacity_mask = token_priority <= self.expert_capacity
        # expert_index = expert_index * expert_capacity_mask

        router_probs = router_probs[expert_index.bool()].reshape(router_probs.shape[: -1] + (1, )) # router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

def Original(args, config) :
    return [None] * config.num_layers

def hypergeom_hypothesis_testing(countings : torch.Tensor) :
    M = 3072 * 16
    N = countings.sum()
    n = 3072
    rv = stats.hypergeom(M, n, N)
    l, r = 0, n
    while l < r :
        mid = (l + r) // 2
        if 1.0 - rv.cdf(mid) <= 0.001 :
            r = mid
        else :
            l = mid + 1
    threshold = l
    assert(1.0 - rv.cdf(threshold) <= 0.001)
    assert(1.0 - rv.cdf(threshold - 1) > 0.001)
    return countings > threshold    

def Function(args, config) :
    selected_experts = []
    for layer in range(config.num_layers) :
        if layer % config.encoder_sparse_step == 1 and layer >= 9 :
            ap = torch.load("../NeuronPredictivity/predictivity/task/TRAIN{}/switchT5/{}.bin".format(args.dataset, layer))
            ap = sum(ap) / len(ap)
            res = torch.zeros((16, ))
            for neuron in ap.topk(k = int(len(ap) * 0.01)).indices :
                res[neuron // 3072] += 1
            flag = hypergeom_hypothesis_testing(res)
            selected_experts.append(torch.LongTensor([expert for expert in range(16) if flag[expert]]))
            print("layer = {}     experts = {}     res = {}".format(layer, selected_experts[-1], res))
            if len(selected_experts[-1]) == 0 :
                selected_experts[-1] = None
        else :
            selected_experts.append(None)
    return selected_experts

def NoFunction(args, config) :
    selected_experts = []
    for layer in range(config.num_layers) :
        if layer % config.encoder_sparse_step == 1 and layer >= 9 :
            ap = torch.load("../NeuronPredictivity/predictivity/task/TRAIN{}/switchT5/{}.bin".format(args.dataset, layer))
            ap = sum(ap) / len(ap)
            res = torch.zeros((16, ))
            for neuron in ap.topk(k = int(len(ap) * 0.01)).indices :
                res[neuron // 3072] += 1
            flag = hypergeom_hypothesis_testing(res)
            selected_experts.append(torch.LongTensor([expert for expert in range(16) if flag[expert]]))
            print("layer = {}     experts = {}     res = {}".format(layer, selected_experts[-1], res))
            if len(selected_experts[-1]) == 0 :
                selected_experts[-1] = None
            else :
                selected_experts[-1] = torch.LongTensor(random.sample([expert for expert in range(16) if not flag[expert]], len(selected_experts[-1])))
        else :
            selected_experts.append(None)
    return selected_experts

strategy = {
    "Original" : Original,
    "Function" : Function,
    "NoFunction" : NoFunction,
}

def get_params(args, model) :
    selected_experts = strategy[args.strategy](args, model.config)
    params = []

    for layer, experts in enumerate(selected_experts) :
        mlp = model.encoder.block[layer].layer[-1].mlp # class SwitchTransformersSparseMLP
        if experts is not None :
            assert layer % model.config.encoder_sparse_step == 1
            new_router = biased_Top1Router(model.config, experts)
            if torch.cuda.is_available() :
                new_router = new_router.cuda()
            new_router.load_state_dict(mlp.router.state_dict())
            mlp.router = new_router
        
        if layer < 8 :
            continue
        
        if layer % 2 == 0 :
            params += list(mlp.wo.parameters())
            pass
        else :
            params += list(mlp.router.parameters())
            for expert in mlp.experts.values() :
                params += list(expert.wo.parameters())
    
    return params

label2token = {
    'sst2' : torch.LongTensor([9412, 248]),
    'mrpc' : torch.LongTensor([465, 2163]),
    'cola' : torch.LongTensor([12153, 2024]),
    'qqp' : torch.LongTensor([465, 2163]),
    'rte' : torch.LongTensor([465, 2163]),
    'mnli' : torch.LongTensor([465, 3836, 2163]),
    'qnli' : torch.LongTensor([465, 2163]),
}

def evaluate(args, model, tokenizer, dataset) :
    model.eval()
    Sum_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(dataset["validation"]) :
        sentences, labels = batch["sentence"], batch["label"]
        total += len(sentences)

        encoded_input = tokenizer(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        input_ids, attention_mask = encoded_input.input_ids, encoded_input.attention_mask
        labels = tokenizer(["<extra_id_0>"] * len(sentences), return_tensors = "pt").input_ids
        labels = torch.cat([labels, labels], -1)
        if torch.cuda.is_available() :
            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            
        with torch.no_grad() :
            output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss, logits = output.loss, output.logits[:, 1, label2token[args.dataset]]
            correct += (logits.argmax(dim = -1).cpu() == batch["label"]).sum().item()
            Sum_loss += loss.item()
            if (total // len(sentences)) % 10 == 0 :
                print("\nloss = {}   acc = {} / {} = {}".format(loss.item(), correct, total, correct / total))
    
    print("[validation dataset] Sum_loss = {}   acc = {}".format(Sum_loss, correct / total))
    return correct / total

def train(args, model, tokenizer, dataset, tunable_params) :
    optimizer = torch.optim.Adam(tunable_params, lr = args.lr)

    if args.training_sample != -1 :
        path = "save/{}_{}training/{}".format(args.dataset, args.training_sample, args.strategy)
    else :
        path = "save/{}_full/{}".format(args.dataset, args.strategy)
    os.makedirs(path, exist_ok = True)

    best_acc = 0.0
    total_step = 0

    best_validation = []

    for epoch in range(args.epoch) :
        Sum_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(dataset["training"]) :
            sentences, labels = batch["sentence"], batch["label"]
            total += len(sentences)

            encoded_input = tokenizer(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
            input_ids, attention_mask = encoded_input.input_ids, encoded_input.attention_mask
            labels = label2token[args.dataset][labels].unsqueeze(-1)
            masks = torch.ones_like(labels) * 32099
            labels = torch.cat([masks, labels], -1)
            if torch.cuda.is_available() :
                input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
            
            model.train()
            optimizer.zero_grad()
            output = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss, logits = output.loss, output.logits[:, 1, label2token[args.dataset]]
            correct += (logits.argmax(dim = -1).cpu() == batch["label"]).sum().item()
            Sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (total // len(sentences)) % 10 == 0 :
                print("\nepoch = {} / {}   loss = {}   acc = {} / {} = {}".format(epoch + 1, args.epoch, loss.item(), correct, total, correct / total))

            if total_step % args.eval_step == 0 :
                acc = evaluate(args, model, tokenizer, dataset)
                if acc > best_acc :
                    print("Update")
                    # torch.save(tunable_params, os.path.join(path, "seed_{}.ckpt".format(args.seed)))
                    best_acc = acc
                    with open(os.path.join(path, "result_{}.txt".format(args.seed)), "w") as fout :
                        fout.write("accuracy = {}\n".format(acc))
                        fout.write("epoch = {} / {}   total_step = {}".format(epoch + 1, args.epoch, total_step))
                best_validation.append(best_acc)
            total_step += 1
        
        print("[training dataset] Sum_loss = {}   acc = {}".format(Sum_loss, correct / total))
    torch.save(best_validation, os.path.join(path, "validation_{}.bin".format(args.seed)))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type = str, default = None)

    parser.add_argument("--dataset", type = str, required = True)
    parser.add_argument("--training_sample", type = int, default = -1)
    parser.add_argument("--validation_sample", type = int, default = -1)

    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--epoch", type = int, default = 20)
    parser.add_argument("--lr", type = float, default = 1E-3)
    parser.add_argument("--bs", type = int, default = 4)
    parser.add_argument("--eval_step", type = int, default = 16)

    parser.add_argument("--strategy", type = str, required = True, choices = ("Original", "Function", "NoFunction"))
    args = parser.parse_args()

    set_gpu(args.gpu)
    seed_everything(args.seed)
    
    model, tokenizer = build_backbone()

    dataset = {'training' : load_dataset(args, 'training'), 'validation' : load_dataset(args, 'validation')}
    if args.training_sample != -1 :
        dataset['training'] = random.sample(dataset['training'], args.training_sample)
    if args.validation_sample != -1 :
        dataset['validation'] = random.sample(dataset['validation'], args.validation_sample)
    for split, dataset_split in dataset.items() :
        dataset[split] = torch.utils.data.DataLoader(dataset = dataset_split, batch_size = args.bs, shuffle = True, drop_last = False)
    
    tunable_params = get_params(args, model)

    train(args, model, tokenizer, dataset, tunable_params)