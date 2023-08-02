import os
if __name__ =="__main__":
    os.system("python codet5_main.py --output_dir=./saved_models --model_name=model.bin --do_train --train_data_file=../data/fine_tune_data/assert_mask_new_train.csv --eval_data_file=../data/fine_tune_data/assert_mask_new_val.csv --test_data_file=../data/fine_tune_data/assert_mask_new_test.csv --epochs 75 --encoder_block_size 512 --decoder_block_size 256 --train_batch_size 8 --eval_batch_size 8 --learning_rate 2e-5 --max_grad_norm 1.0 --n_gpu 1 --evaluate_during_training --seed 123456  2>&1 | tee train.log")
    os.system("python codet5_main.py --output_dir=./saved_models --model_name=model.bin --do_test --test_data_file=../data/fine_tune_data/assert_mask_new_test.csv --encoder_block_size 512 --decoder_block_size 256 --eval_batch_size 1 --n_gpu 1")
