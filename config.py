import sys
import argparse

_using_debugger = getattr(sys, "gettrace", None)() is not None

parser = argparse.ArgumentParser(description='Prefix classification task')

parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased', help='bert model name')
parser.add_argument('--batch_size', type=int, default=64, help='number of samples in actual batch')
parser.add_argument('--train_dataset_path', type=str, help='train dataset file path')
parser.add_argument('--dev_dataset_path', type=str, help='validation dataset file path')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to run')
parser.add_argument('--num_features_in_last_layer', type=int, default=768, help='num features in last layer')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--patience', type=float, default=2, help='patience for early stopping')
parser.add_argument('--min_delta', type=float, default=0.05, help='min delta for early stopping')
parser.add_argument('--max_length', type=int, default=512, help='max token length')
parser.add_argument('--seed', type=int, default=43, help='random seed')
parser.add_argument('--log_dir', type=str, default="log", help='log directory')
parser.add_argument('--checkpoint_prefix', type=str, default="", help="checkpoint prefix name")
parser.add_argument('--tensorboard_log_dir', default="log_dir", help='tensorboard logs dir')
parser.add_argument('--mode', default="complete", help='mode for running experiment')
parser.add_argument('--warmup_steps', default=1e4, type=int, help='number of lr warmup steps')
parser.add_argument('--models_dir', type=str, default="models/", help='output model directory')


