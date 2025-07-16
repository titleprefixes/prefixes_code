from transformers import BertModel, BertTokenizerFast
from model import FineTuneBertMultiClassCLS
import torch
from transformers import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random
import os
import math
import pandas as pd

NO_DECAY = ["LayerNorm", "layer_norm", "bias"]


def setup_optimizer_params(model, weight_decay=0.0):
    """
    """
    # select parameters we want to finetune
    optimizer_params = []
    for name, params in model.named_parameters():
        optimizer_params.append((name, params))

    # if we want to use a weight decay, we don't apply it to biases and layer norm params
    with_decay = {
        'params': [params for name, params in optimizer_params
                   if not any(no_decay in name for no_decay in NO_DECAY)],
        'weight_decay': weight_decay
    }
    without_decay = {
        'params': [params for name, params in optimizer_params
                   if any(no_decay in name for no_decay in NO_DECAY)],
        'weight_decay': 0.0
    }
    return [with_decay, without_decay]


def get_subsets(tokenizer, row):
    result = {}
    title = row['title']
    tokens = tokenizer(title)
    max_len = math.floor(len(tokens) / 2)

    for i in range(1, max_len + 1):
        sub_title = tokens[:i].text
        sub_res = {key: val for key, val in row.items() if key != 'title' and key != 'asin'}
        sub_res['asin'] = str(row['asin']) + "_" + str(i)
        sub_res['title'] = sub_title
        result[i] = sub_res
    return result


def create_subsets_df(df, tokenizer):
    result = {}
    running_index = 0
    for i, row in df.iterrows():
        sub_res = get_subsets(tokenizer, row)
        for key in sub_res:
            result[running_index] = sub_res[key]
            running_index += 1
    subsets_df = pd.DataFrame.from_dict(result, orient='index')
    return subsets_df


def process_df(df, tokenizer):
    df['tok_len'] = df['title'].apply(lambda x: len(tokenizer(x)))
    df = df[df.label != 102]
    df = df[df.tok_len >= 2].reset_index(drop=True)
    df['label'] = pd.Categorical(df['label'])
    df['label'] = df['label'].cat.codes
    return df


def save_ckpt(model, model_dir, name):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + "/" + name + ".pth")


def load_pretrained_bert(path_to_pretrained, *args):
    return BertModel \
        .from_pretrained(path_to_pretrained)


def load_tokenizer_bert(*args):
    return BertTokenizerFast.from_pretrained(args[0])


def load_tokenizer_bert_no_lower(*args):
    return BertTokenizerFast.from_pretrained(args[0], do_lower_case=False)


#

def get_out_sz_bert(bert, *args):
    return bert.pooler.dense.out_features


def model_and_tokenizer_from_spec_cls(model_name, num_classes=2):
    print('loading pretrained model_utils...')
    bert = load_pretrained_bert(model_name)
    print('loading tokenizer...')
    if model_name.__contains__("uncased"):
        tokenizer = load_tokenizer_bert(model_name)
    else:
        tokenizer = load_tokenizer_bert_no_lower(model_name)
    print(f'Creating fine-tuning architecture for multiclass classification (Num Classes: {num_classes})')
    num_emb_features_fcn = get_out_sz_bert(bert)
    model = FineTuneBertMultiClassCLS(bert, num_emb_features_fcn, num_classes=num_classes)
    return model, tokenizer

def init_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model_huggingface(model, train_dl, dev_dl, learning_rate, wd, epochs, warmup_steps, seed, log_dir, models_dir,
                            model_save_name, patience, min_delta):


    PATIENCE = patience
    BEST_ACC = 0
    MIN_DELTA = min_delta
    BEST_LOSS = float("inf")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criteria = torch.nn.CrossEntropyLoss()

    total_steps = len(train_dl) * epochs

    writer = SummaryWriter(log_dir=log_dir + "/" + model_save_name + "/")

    parameter_groups = setup_optimizer_params(model, wd)
    optimizer = AdamW(parameter_groups, lr=learning_rate, eps=1e-6)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    global_step_train = 0
    termination = False
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...', flush=True)
        if termination:
            print("No improvement in validation loss, terminating")
            break

        total_train_loss = 0
        total_bs_train = 0

        model.train()

        for step, batch in enumerate(tqdm(train_dl)):
            inp, labels = batch
            model_input = tuple([item.to(device) for item in inp])
            labels = labels.to(device)

            outputs = model(model_input)
            loss = criteria(outputs, labels)

            total_train_loss += (loss.item() * len(labels))
            total_bs_train += len(labels)

            writer.add_scalar("Loss/train", loss.item(), global_step_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step_train += 1
            # accumulating_example_count += batch_size

        avg_train_loss = total_train_loss / total_bs_train

        print("")
        print("  Average training loss: {}".format(avg_train_loss), flush=True)

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        model.eval()

        total_eval_loss = 0
        total_bs = 0
        all_labels = []
        all_preds = []
        # Evaluate data for one epoch
        for batch in dev_dl:
            with torch.no_grad():
                inp, labels = batch
                model_input = tuple([item.to(device) for item in inp])
                labels = labels.to(device)

                outputs = model(model_input)
                loss = criteria(outputs, labels)
                total_eval_loss += (loss.item() * len(labels))
                total_bs += len(labels)

                predicted_labels = outputs.argmax(dim=-1)
                all_preds.extend(list(predicted_labels.cpu().numpy()))
                all_labels.extend(list(labels.cpu().numpy()))
        avg_val_loss = total_eval_loss / total_bs

        writer.add_scalar("Loss/dev", avg_val_loss, epoch_i)
        acc = np.mean([1 if i == j else 0 for i, j in zip(all_preds, all_labels)])
        writer.add_scalar("Accuracy/dev", acc, epoch_i)

        print("  Validation Loss: {}".format(avg_val_loss), flush=True)
        print("  Validation classification accuracy: {}".format(acc), flush=True)

        """Early Stopping"""
        if epoch_i == 0:
            BEST_LOSS = avg_val_loss
            BEST_ACC = acc
            save_ckpt(model, models_dir, model_save_name + "_best_accuracy")
            continue
        else:
            if acc > BEST_ACC:
                BEST_ACC = acc
                save_ckpt(model, models_dir, model_save_name + "_best_accuracy")
            if BEST_LOSS - MIN_DELTA <= avg_val_loss:
                PATIENCE -= 1
                if PATIENCE == 0:
                    termination = True
            else:
                PATIENCE = patience
                BEST_LOSS = avg_val_loss

    save_ckpt(model, models_dir, model_save_name + "_final")

    print(f'FINAL ACCURACY {acc}')
    print(f'BEST ACCURACY {BEST_ACC}')
    writer.add_scalar("Accuracy/dev_best_accuracy", BEST_ACC, 0)
    writer.close()




def train_model_lstm(model,
                     models_dir,
                     checkpoint_name,
                     train_loader,
                     valid_loader,
                     optimizer,
                     num_epochs=10,
                     device=torch.device('cuda'),
                     log_dir="./",
                     min_delta=0.05,
                     patience=2,
                     warmup_steps=1e4
                     ):
    # LSTM_MODEL_DIR = "lstm_models/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    MIN_DELTA = min_delta
    PATIENCE = patience

    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0
    termination = False

    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, checkpoint_name))
    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # training loop
    for epoch_i in range(num_epochs):
        model.train()

        if termination:
            print("No improvement in validation loss, terminating")
            break

        total_train_loss = 0
        total_bs_train = 0
        for batch in tqdm(train_loader):
            titles, titles_len, labels = batch
            labels = labels.to(device)
            titles = titles.to(device)
            titles_len = titles_len.to(device)
            outputs = model(titles, titles_len)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss.item(), global_step)
            total_train_loss += (loss.item() * len(labels))
            total_bs_train += len(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # update running values
            global_step += 1

        avg_train_loss = total_train_loss / total_bs_train
        print("")
        print("  Average training loss: {}".format(avg_train_loss), flush=True)

        print("")
        print("Running Validation...")

        total_eval_loss = 0
        total_bs = 0
        all_labels = []
        all_preds = []
        # Evaluate data for one epoch
        model.eval()

        for batch in valid_loader:
            with torch.no_grad():
                titles, titles_len, labels = batch
                titles = titles.to(device)
                titles_len = titles_len.to(device)
                labels = labels.to(device)
                outputs = model(titles, titles_len)
                loss = criterion(outputs, labels)
                total_eval_loss += (loss.item() * len(labels))
                total_bs += len(labels)

                predicted_labels = outputs.argmax(dim=-1)
                all_preds.extend(list(predicted_labels.cpu().numpy()))
                all_labels.extend(list(labels.cpu().numpy()))
        avg_val_loss = total_eval_loss / total_bs
        writer.add_scalar("Loss/dev", avg_val_loss, epoch_i)
        acc = np.mean([1 if i == j else 0 for i, j in zip(all_preds, all_labels)])
        writer.add_scalar("Accuracy/dev", acc, epoch_i)
        print("  Validation Loss: {}".format(avg_val_loss), flush=True)
        print("  Validation classification accuracy: {}".format(acc), flush=True)

        if epoch_i == 0:
            BEST = avg_val_loss
            LOWEST_LOSS = avg_val_loss
            BEST_ACC = acc
            save_ckpt(model, models_dir, checkpoint_name + "_best_accuracy")
            continue
        else:
            if BEST_ACC < acc:
                save_ckpt(model, models_dir, checkpoint_name + "_best_accuracy")
                BEST_ACC = acc
            if avg_val_loss < LOWEST_LOSS:
                LOWEST_LOSS = avg_val_loss

            if BEST - MIN_DELTA <= avg_val_loss:
                PATIENCE -= 1
                if PATIENCE == 0:
                    termination = True
            else:
                PATIENCE = 2
                BEST = avg_val_loss
    writer.add_scalar("Accuracy/dev_best_accuracy", BEST_ACC, 0)
    writer.close()
    save_ckpt(model, models_dir, checkpoint_name + "_final")
    print('Finished Training!')
    print("BEST ACC", BEST_ACC)
