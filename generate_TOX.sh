python ./scripts/sample.py --model_load $1 --target TOX & wait
python ./eval/01_result2input.py --target TOX & wait
python ./Toxicity_predictor/eval_tox.py --data_path ./eval/results/TOX_tmp.csv --checkpoint_dir ./Toxicity_predictor/ckpt/ & wait
mv ./predict_TOX.csv ./eval/results/ & wait
python ./eval/02_eval.py --target TOX & wait