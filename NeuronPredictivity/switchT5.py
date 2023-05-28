import types
import torch
import numpy as np
import transformers

def _forward(ffn_self, hidden_states) :
    bsz, seq_len, hidden_size = hidden_states.shape
    hidden_states_mlp = hidden_states.clone().detach()
    hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)

    score = ffn_self.wr(hidden_states_mlp)
    score = torch.nn.functional.softmax(score, dim = -1)

    weight, labels = torch.topk(score, k = 1, dim = -1)
    labels, weight = labels.view(bsz, seq_len, 1), weight.view(bsz, seq_len, 1)
    cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
    
    hidden_states = ffn_self.wi(hidden_states)
    hidden_states = torch.nn.functional.relu(hidden_states)
    hidden_states[cur_mask == False] = 0  
    hidden_states = ffn_self.dropout(hidden_states)
    hidden_states = ffn_self.wo(hidden_states)
    hidden_states = hidden_states * weight
    return hidden_states

def get_switchT5(device, checkpoint : str, tie_word_embeddings : bool = None) :
    config = transformers.T5Config.from_pretrained("t5-base")
    if tie_word_embeddings is not None :
        config.tie_word_embeddings = tie_word_embeddings
    model = transformers.T5EncoderModel.from_pretrained("t5-base", config = config)
    expert_num = 16

    for idx, layer in enumerate(model.encoder.block) :
        if idx % 2 == 1 :
            layer.layer[1].DenseReluDense.wi.weight.data = torch.cat([torch.zeros_like(layer.layer[1].DenseReluDense.wi.weight)] * expert_num, 0)
            layer.layer[1].DenseReluDense.wo.weight.data = torch.cat([torch.zeros_like(layer.layer[1].DenseReluDense.wo.weight)] * expert_num, -1)
            layer.layer[1].DenseReluDense.wr = torch.nn.Linear(768, expert_num, bias = False)

            patterns = []
            for i in range(expert_num) :
                tmp = np.array([0] * (3072 * expert_num))
                tmp[i * 3072 : (i + 1) * 3072] = 1
                patterns.append(tmp)
            layer.layer[1].DenseReluDense.patterns = torch.Tensor(patterns).to(device)
            layer.layer[1].DenseReluDense.forward_old = layer.layer[1].DenseReluDense.forward
            layer.layer[1].DenseReluDense.forward = types.MethodType(_forward, layer.layer[1].DenseReluDense)
    
    matched_params = torch.load(checkpoint)
    names = [key for key, value in model.named_parameters()] + ["encoder.embed_tokens.weight"]
    model.load_state_dict({name : torch.tensor(matched_params[name]) for name in matched_params if name in names}, strict = True)
    model = model.to(device)
    return model