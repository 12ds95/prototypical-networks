python scripts/train/few_shot/run_train.py --data.way 60 --data.shot 5 --data.test_shot 5 --data.test_way 20 --log.exp_dir=results/60_20way5shot --data.cuda 
python scripts/predict/few_shot/run_eval.py --data.test_shot 5 --data.test_way 20 --model.model_path=results/60_20way5shot/best_model.t7
