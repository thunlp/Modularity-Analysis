# Modularity Analysis

Source codes and datasets for *[Emergent Modularity in Pre-trained Transformers](https://arxiv.org/abs/2305.18390)*.

## Neuron Predictivity Calculation

Download the checkpoints and evaluation data.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/NeuronPredictivityData.tar
tar -xvf NeuronPredictivityData.tar
rm -r NeuronPredictivityData.tar

wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/NeuronPredictivityCheckpoint.tar
tar -xvf NeuronPredictivityCheckpoint.tar
rm -r NeuronPredictivityCheckpoint.tar
```

Then run `cd NeuronPredictivity && bash scripts.sh` to get the neuron predictivity.

Or directly download the neuron predictivity.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/NeuronPredictivityPred.tar
tar -xvf NeuronPredictivityPred.tar
rm -r NeuronPredictivityPred.tar
```

## Functional Specialization of Neurons


At each transformer layer, we compute the best predictivity of neurons for each sub-function and then calculate the average best predictivity among all sub-function in each function. We can get the figures (Figure 1 Left) by the following commands.

```bash
cd NeuronFunctionalSpecialization

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
cd NeuronFunctionalSpecialization

python Figure1Right_maker.py T5
python Figure1Right_maker.py switchT5

# We already put the results in the folder.
```

## Functional Expert


We first get the neurons with the highest 1% predictivities for each sub-function.

```bash
cd FunctionalExpert

python TopK_neurons.py --model T5
python TopK_neurons.py --model switchT5
python TopK_neurons_step.py --model T5_step
python TopK_neurons_step.py --model switchT5_step
```

Or directly download the result.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/FunctionalExpertTopKNeurons.tar
tar -xvf FunctionalExpertTopKNeurons.tar
rm -r FunctionalExpertTopKNeurons.tar
```

Then we conduct hypothesis testing to get the functional experts.

```bash
cd FunctionalExpert

python HypothesisTesting.py --model T5
python HypothesisTesting.py --model switchT5
python HypothesisTesting_step.py --model T5_step
python HypothesisTesting_step.py --model switchT5_step

python HypothesisTesting4RandomPartitioning.py --model T5
python HypothesisTesting4RandomPartitioning.py --model switchT5
```

Or directly download the result.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/FunctionalExpertHypothesisTesting.tar
tar -xvf FunctionalExpertHypothesisTesting.tar
rm -r FunctionalExpertHypothesisTesting.tar
```

Finally, we run `cd FunctionalExpert && python Table1and4_maker.py` to get the numbers in Table 1 and Table 4.

## Perturbation Experiments for T5

```bash
cd Perturbation4T5
```

We first run `python average_predictivity.py` to get the average predictivities on all datasets (we already put the results in the folder). We then run `python run_perturbation.py` to conduct the perturbation experiments for T5 (we already put the results in the folder). Finally, we run `python Figure2_maker.py` to get the figures (Figure 2).

## Perturbation Experiments for Switch Transformer

```bash
cd Perturbation4SwitchTransformer
```

We run `bash scripts.sh` to conduct the perturbation experiments for Switch Transformer (we already put the results in the folder). Finally, we run `python Table3_maker.py` to get the numbers in Table 3.

## Stabilization during Pre-training

```bash
cd Stabilization
python ExpertPredictivity.py --model T5_step
python ExpertPredictivity.py --model switchT5_step
```

Or directly download the result.

```bash
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/StabilizationExpertPredictivity.tar
tar -xvf StabilizationExpertPredictivity.tar
rm -r StabilizationExpertPredictivity.tar
```

Then calculate the the similarity between a layer of two model checkpoints w.r.t. a particular sub-function, at the expert/neuron level.

```bash
cd Stabilization
python ExpertAndNeuron_Stabilization.py --model T5_step
python ExpertAndNeuron_Stabilization.py --model switchT5_step
python RandomPartitioning_Stabilization.py --model T5_step       --epoch 50
python RandomPartitioning_Stabilization.py --model switchT5_step --epoch 30

# We already put the results in the folder.
```

Finally we can get the figures (Figure 4) by the following commands.

```bash
cd Stabilization
python Figure4_maker.py --model T5_step
python Figure4_maker.py --model switchT5_step

# We already put the results in the folder.
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
