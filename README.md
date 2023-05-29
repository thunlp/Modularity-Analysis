# Modularity Analysis

Source code for "Emergent Modularity in Pre-trained Transformers".

We will update all datasets and checkpoints in this week.

## Neuron Predictivity Calculation

```bash
cd NeuronPredictivity
```

Download the checkpoints and evaluation data.

```bash
# TODO
# NeuronPredictivity/checkpoint
# NeuronPredictivity/data
```

Then run the jobs `bash scripts.sh`.

Or directly download the predictivity.

```bash
# TODO
# NeuronPredictivity/predictivity
```

## Functional Specialization of Neurons

```bash
cd NeuronFunctionalSpecialization
```

At each transformer layer, we compute the best predictivity of neurons for each sub-function and then calculate the average best predictivity among all sub-function in each function. We can get the figures (Figure 1 Left) by the following commands.

```bash
python Figure1Left_maker.py switchT5 semantic
python Figure1Left_maker.py switchT5 knowledge
python Figure1Left_maker.py switchT5 task

python Figure1Left_maker.py T5 semantic
python Figure1Left_maker.py T5 knowledge
python Figure1Left_maker.py T5 task

# We already put the results in the folder.
```

We also calculate the average distribution similarity between different functions of different layers for the same pre-trained models. We can get the figures (Figure 1 Right) by the following commands.

```bash
python Figure1Right_maker.py T5
python Figure1Right_maker.py switchT5

# We already put the results in the folder.
```

## Functional Expert

```bash
cd FunctionalExpert
```

We first get the neurons with the highest 1% predictivities for each sub-function.

```bash
python TopK_neurons.py --model T5
python TopK_neurons.py --model switchT5
python TopK_neurons_step.py --model T5_step
python TopK_neurons_step.py --model switchT5_step
```

Or directly download the result.

```bash
# TODO
# FunctionalExpert/TopK_neurons
```

Then we conduct hypothesis testing to get the functional experts.

```bash
python HypothesisTesting.py --model T5
python HypothesisTesting.py --model switchT5
python HypothesisTesting_step.py --model T5_step
python HypothesisTesting_step.py --model switchT5_step

python HypothesisTesting4RandomPartitioning.py --model T5
python HypothesisTesting4RandomPartitioning.py --model switchT5
```

Or directly download the result.

```bash
# TODO
# FunctionalExpert/HypothesisTesting
```

Finally, we run `Table1and4_maker.py` to get the numbers in Table 1 and Table 4.

## Perturbation Experiments for T5

```bash
cd Perturbation4T5
```

We first run `python average_predictivity.py` to get the average predictivities on all datasets (we already put the results in the folder). We then run `run_perturbation.py` to conduct the perturbation experiments for T5 (we already put the results in the folder). Finally, we run `Figure2_maker.py` to get the figures (Figure 2).

## Perturbation Experiments for Switch Transformer

```bash
cd Perturbation4SwitchTransformer
```

We run `bash scripts.sh` to conduct the perturbation experiments for Switch Transformer (we already put the results in the folder). Finally, we run `Table3_maker.py` to get the numbers in Table 3.

## Stabilization during Pre-training

```bash
cd Stabilization
```

```bash
python ExpertPredictivity.py --model T5_step
python ExpertPredictivity.py --model switchT5_step
```

Or directly download the result.

```bash
# TODO
# Stabilization/ExpertPredictivity
```

Then calculate the the similarity between a layer of two model checkpoints w.r.t. a particular sub-function, at the expert/neuron level.

```bash
python ExpertAndNeuron_Stabilization.py --model T5_step
python ExpertAndNeuron_Stabilization.py --model switchT5_step
python RandomPartitioning_Stabilization.py --model T5_step       --epoch 50
python RandomPartitioning_Stabilization.py --model switchT5_step --epoch 30
```

We already put the results in the folder.

Finally we can get the figures (Figure 4) by the following commands.

```bash
python Figure4_maker.py --model T5_step
python Figure4_maker.py --model switchT5_step
```

## Cite

If you use the code, please cite this paper:

```
@inproceedings{zhang2023emergent,
  title={Emergent Modularity in Pre-trained Transformers},
  author={Zhang, Zhengyan and Zeng, Zhiyuan and Lin, Yankai and Xiao, Chaojun and Wang, Xiaozhi and Han, Xu and Liu, Zhiyuan and Xie, Ruobing and Sun, Maosong and Zhou, Jie},
  booktitle={Findings of ACL},
  year={2023}
}
```
