python ./scripts/sample.py --model_load $1 --target SA & wait
python ./eval/01_result2input.py --target SA & wait
python ./eval/02_eval.py --target SA & wait