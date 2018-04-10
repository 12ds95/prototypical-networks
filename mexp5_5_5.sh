python scripts/train/few_shot/run_train.py --train.learning_rate 0.0000001 --data.dataset=miniImagenet --model.x_dim=3,84,84 --data.way 5 --data.shot 5 --data.test_shot 5 --data.test_way 5 --log.exp_dir=results/m5_5way5shot 
python scripts/predict/few_shot/run_eval.py --data.test_shot 5 --data.test_way 5 --model.model_path=results/m5_5way5shot/best_model.t7
