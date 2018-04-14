python scripts/train/few_shot/run_train.py --data.way 5 --data.shot 5 --data.test_shot 5 --data.test_way 5 --data.cuda --log.exp_dir=results/5_5way5shot 
python scripts/predict/few_shot/run_eval.py --data.test_shot 5 --data.test_way 5 --model.model_path=results/5_5way5shot/best_model.t7
