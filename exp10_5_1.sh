python scripts/train/few_shot/run_train.py --data.way 10 --data.shot 1 --data.test_shot 1 --data.test_way 5 --data.cuda --log.exp_dir=results/10_5way1shot 
python scripts/predict/few_shot/run_eval.py --data.test_shot 1 --data.test_way 5 --model.model_path=results/10_5way1shot/best_model.t7
