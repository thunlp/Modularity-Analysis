for seed in {0..7}
do
    python run.py --dataset cola --training_sample 128 --strategy Original       --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset cola --training_sample 128 --strategy Function   --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset cola --training_sample 128 --strategy NoFunction --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset mnli --training_sample 128 --validation_sample 1000 --strategy Original       --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset mnli --training_sample 128 --validation_sample 1000 --strategy Function   --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset mnli --training_sample 128 --validation_sample 1000 --strategy NoFunction --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset mrpc --training_sample 128 --strategy Original       --epoch 50 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset mrpc --training_sample 128 --strategy Function   --epoch 50 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset mrpc --training_sample 128 --strategy NoFunction --epoch 50 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset rte --training_sample 128 --strategy Original       --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset rte --training_sample 128 --strategy Function   --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset rte --training_sample 128 --strategy NoFunction --epoch 100 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset qnli --training_sample 128 --validation_sample 1000 --strategy Original       --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset qnli --training_sample 128 --validation_sample 1000 --strategy Function   --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset qnli --training_sample 128 --validation_sample 1000 --strategy NoFunction --epoch 300 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset qqp --training_sample 128 --validation_sample 1000 --strategy Original       --epoch 150 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset qqp --training_sample 128 --validation_sample 1000 --strategy Function   --epoch 150 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset qqp --training_sample 128 --validation_sample 1000 --strategy NoFunction --epoch 150 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0

    python run.py --dataset sst2 --training_sample 128 --strategy Original       --epoch 200 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset sst2 --training_sample 128 --strategy Function   --epoch 200 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
    python run.py --dataset sst2 --training_sample 128 --strategy NoFunction --epoch 200 --lr 2E-4 --bs 16 --eval_step 4 --seed $seed --gpu 0
done