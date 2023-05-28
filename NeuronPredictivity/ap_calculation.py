import torch
from sklearn.metrics import average_precision_score
from sklearn import preprocessing

def task_CalAP(dataset, task, acts) :
    def CalAP(positive : tuple, negative : tuple) :
        vecs, labels = [], []
        for label in positive :
            for instance in dataset[label] :
                vecs.append(acts[instance["idx"]])
                labels.append(0)
        for label in negative :
            for instance in dataset[label] :
                vecs.append(acts[instance["idx"]])
                labels.append(1)

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels_inv = 1 - labels

        vecs = torch.stack(vecs)
        length = vecs.shape[-1]
        
        return torch.tensor([max(average_precision_score(labels, vecs[:, i]), 
                                average_precision_score(labels_inv, vecs[:, i]))
                                for i in range(length)])
    if task.startswith("TRAIN") :
        task = task[len("TRAIN") :]
    if task == "cola" :
        return (CalAP(("0", ), ("1", )), )
    if task == "rte" :
        return (CalAP(("0", ), ("1", )), )
    if task == "mnli" :
        return (CalAP(("0", "2"), ("1", )), CalAP(("0", ), ("2", )))
    if task == "mrpc" :
        return (CalAP(("0", ), ("1", )), )
    if task == "qnli" :
        return (CalAP(("0", ), ("1", )), )
    if task == "qqp" :
        return (CalAP(("0", ), ("1", )), )
    if task == "sst2" :
        return (CalAP(("0", ), ("1", )), )
    if task == "imdb" :
        return (CalAP(("0", ), ("1", )), )
    if task == "tweet" :
        return (CalAP(("0", "2"), ("1", )), CalAP(("0", ), ("2", )))
    assert(False)

def knowledge_CalAP(acts, Filter = lambda key : True) :
    relation_set = set()
    with open("data/knowledge/extracted_triples.txt", "r") as fin :
        for line in fin :
            relation_set.add(line.split("\t")[0])
    relation_ap = {}
    for relation in relation_set :
        if not Filter(relation) :
            continue
        vecs, labels = [], []
        with open("data/knowledge/extracted_triples.txt", "r") as fin :
            index = 0
            for line in fin.readlines() :
                rel = line.strip().split("\t")[0]
                label = line.strip().split("\t")[1]
                if Filter(rel) :
                    if rel == relation and line.strip().split("\t")[2] == "0" :
                        vecs.append(acts[index])
                        labels.append(label)
                    index += 1
            assert(index == len(acts))

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels_inv = 1 - labels

        vecs = torch.stack(vecs)
        length = vecs.shape[-1]
        relation_ap[relation] = torch.tensor([max(average_precision_score(labels, vecs[:, i]), 
                                                    average_precision_score(labels_inv, vecs[:, i]))
                                                    for i in range(length)])
    return relation_ap

def semantic_CalAP(acts, Filter = lambda key : True) :
    sense_map = {}
    with open("data/semantic/target_samples.txt", "r") as fin :
        for line in fin :
            id, sense = line.strip().split(" ")
            sense_map[id] = sense
    word_ap = {}
    with open("data/semantic/sense_pairs.txt", "r") as fin :
        sense_pairs = fin.readlines()
    for word in sense_pairs :
        word = word.strip().split(" ")[0]
        if not Filter(word) :
            continue
        vecs, labels = [], []
        with open("data/semantic/extracted_samples.txt", "r") as fin :
            index = 0
            for line in fin.readlines() :
                sense = sense_map[line.strip().split("\t")[0]]
                cur_word = sense.split("%")[0]
                if Filter(cur_word) :
                    if cur_word == word :
                        vecs.append(acts[index])
                        labels.append(sense)
                    index += 1
            assert(index == len(acts))

        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        labels_inv = 1 - labels

        vecs = torch.stack(vecs)
        length = vecs.shape[-1]
        word_ap[word] = torch.tensor([max(average_precision_score(labels, vecs[:, i]), 
                                            average_precision_score(labels_inv, vecs[:, i]))
                                            for i in range(length)])
    return word_ap