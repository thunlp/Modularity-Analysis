import torch

class NeuronProcessor :
    def __init__(self) :
        pass
    
    def observe(self, model, result, attention_mask) :
        def forward_observe(layer) :
            def fn(intermediate, input, output : torch.Tensor) :
                output[attention_mask == 0, :] = -1E12
                output = torch.nn.functional.relu(output) # pass a RELU activation function
                result[layer].append(output.max(dim = 1).values.cpu())
            return fn
        hooks = []
        for layer in range(12) :
            intermediate = model.encoder.block[layer].layer[1].DenseReluDense.wi # before RELU
            hook = intermediate.register_forward_hook(forward_observe(layer))
            hooks.append(hook)
        return hooks
    
    def erase_hooks(self, hooks) :
        for hook in hooks :
            hook.remove()