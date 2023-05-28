python run.py --function semantic --model T5        --gpu 0
python run.py --function semantic --model T5        --gpu 0 --initialization random
python run.py --function semantic --model switchT5  --gpu 0
python run_step.py --function semantic --model T5_step        --gpu 0
python run_step.py --function semantic --model switchT5_step  --gpu 0


python run.py --function knowledge --model T5        --gpu 0
python run.py --function knowledge --model T5        --gpu 0 --initialization random
python run.py --function knowledge --model switchT5  --gpu 0
python run_step.py --function knowledge --model T5_step        --gpu 0
python run_step.py --function knowledge --model switchT5_step  --gpu 0


python run.py --function TASKcola --model T5        --gpu 0
python run.py --function TASKcola --model T5        --gpu 0 --initialization random
python run.py --function TASKcola --model switchT5  --gpu 0
python run_step.py --function TASKcola --model T5_step        --gpu 0
python run_step.py --function TASKcola --model switchT5_step  --gpu 0

python run.py --function TASKmnli --model T5        --gpu 0
python run.py --function TASKmnli --model T5        --gpu 0 --initialization random
python run.py --function TASKmnli --model switchT5  --gpu 0
python run_step.py --function TASKmnli --model T5_step        --gpu 0
python run_step.py --function TASKmnli --model switchT5_step  --gpu 0

python run.py --function TASKmrpc --model T5        --gpu 0
python run.py --function TASKmrpc --model T5        --gpu 0 --initialization random
python run.py --function TASKmrpc --model switchT5  --gpu 0
python run_step.py --function TASKmrpc --model T5_step        --gpu 0
python run_step.py --function TASKmrpc --model switchT5_step  --gpu 0

python run.py --function TASKqnli --model T5        --gpu 0
python run.py --function TASKqnli --model T5        --gpu 0 --initialization random
python run.py --function TASKqnli --model switchT5  --gpu 0
python run_step.py --function TASKqnli --model T5_step        --gpu 0
python run_step.py --function TASKqnli --model switchT5_step  --gpu 0

python run.py --function TASKqqp --model T5        --gpu 0
python run.py --function TASKqqp --model T5        --gpu 0 --initialization random
python run.py --function TASKqqp --model switchT5  --gpu 0
python run_step.py --function TASKqqp --model T5_step        --gpu 0
python run_step.py --function TASKqqp --model switchT5_step  --gpu 0

python run.py --function TASKrte --model T5        --gpu 0
python run.py --function TASKrte --model T5        --gpu 0 --initialization random
python run.py --function TASKrte --model switchT5  --gpu 0
python run_step.py --function TASKrte --model T5_step        --gpu 0
python run_step.py --function TASKrte --model switchT5_step  --gpu 0

python run.py --function TASKsst2 --model T5        --gpu 0
python run.py --function TASKsst2 --model T5        --gpu 0 --initialization random
python run.py --function TASKsst2 --model switchT5  --gpu 0
python run_step.py --function TASKsst2 --model T5_step        --gpu 0
python run_step.py --function TASKsst2 --model switchT5_step  --gpu 0