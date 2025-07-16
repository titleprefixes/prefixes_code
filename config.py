import sys
import argparse

_using_debugger = getattr(sys, "gettrace", None)() is not None

parser = argparse.ArgumentParser(description='Prefix classification task')

parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased', help='bert model name')
parser.add_argument('--batch_size', type=int, default=64, help='number of samples in actual batch')
parser.add_argument('--train_dataset_path', type=str, help='train dataset file path')
parser.add_argument('--dev_dataset_path', type=str, help='validation dataset file path')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to run')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--patience', type=float, default=2, help='patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.05, help='min delta for early stopping')
parser.add_argument('--max_length', type=int, default=512, help='max token length')
parser.add_argument('--seed', type=int, default=9001, help='random seed')
parser.add_argument('--log_dir', type=str, default="log", help='log directory')
parser.add_argument('--checkpoint_prefix', type=str, default="model_", help="checkpoint prefix name")
parser.add_argument('--tensorboard_log_dir', default="log_dir", help='tensorboard logs dir')
parser.add_argument('--mode', default="complete", help='mode for running experiment')
parser.add_argument('--warmup_steps', default=1e4, type=int, help='number of lr warmup steps')
parser.add_argument('--models_dir', type=str, default="models/", help='output model directory')
parser.add_argument('--w2v_model_dir', type=str, default="w2v_models/", help='w2v model path base directory')
parser.add_argument('--w2v_model_path_prefix', type=str, default="w2v_model", help='w2v model path prefix')
parser.add_argument('--w2v_input_path', type=str, default="w2v_clean_data.csv", help='w2v input file path for training')
parser.add_argument('--w2v_window_size', type=int, default=4, help='window size for w2v training')


