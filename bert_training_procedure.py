from dataset import collate_batch_amazon, get_data_loader_amazon_bert
from functools import partial
from utils import model_and_tokenizer_from_spec_cls, train_model_huggingface, process_df, create_subsets_df, \
    init_random_seed
from config import parser
import pandas as pd
import spacy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    hparams = parser.parse_args()
    init_random_seed(hparams.seed)

    train = pd.read_csv(hparams.train_dataset_path,
                        keep_default_na=False, na_values=['_'])
    dev = pd.read_csv(hparams.dev_dataset_path,
                      keep_default_na=False, na_values=['_'])

    if hparams.mode in ['subsets', 'random']:
        spacy_tokenizer = spacy.blank("en") # If you wish to change the tokenizer you will have to change the \
        # definition inside the dataset classes as well (defined this way due to spacy's instability in multithreading)
        train['title'] = train['title'].apply(lambda x: " ".join(x.split()))
        dev['title'] = dev['title'].apply(lambda x: " ".join(x.split()))

        # removes class 102 and reassings label for alignment between classes in amazon dataset
        train = process_df(train, spacy_tokenizer)
        dev = process_df(dev, spacy_tokenizer)

        if hparams.mode == 'subsets':
            # extracting all prefix subsets from original data
            train = create_subsets_df(train, spacy_tokenizer)
            dev = create_subsets_df(dev, spacy_tokenizer)

    model, tokenizer = model_and_tokenizer_from_spec_cls(hparams.bert_model_name, num_classes=train['label'].nunique())
    collate_fn = partial(collate_batch_amazon, pad_token_id=tokenizer.pad_token_id)

    train_dl = get_data_loader_amazon_bert(train, tokenizer, collate_fn, hparams.mode,
                                           batch_size=hparams.batch_size,
                                           num_workers=4,
                                           shuffle=True)
    dev_dl = get_data_loader_amazon_bert(dev, tokenizer, collate_fn, hparams.mode,
                                         batch_size=hparams.batch_size,
                                         num_workers=4)

    checkpoint_name = hparams.checkpoint_prefix + "_".join(
        [str(hparams.seed), str(hparams.batch_size), str(hparams.wd), str(hparams.lr), str(hparams.num_epochs), hparams.mode])

    # conduct training with given parameters
    train_model_huggingface(model, train_dl, dev_dl, hparams.lr, hparams.wd, hparams.num_epochs,
                            hparams.warmup_steps, hparams.seed, hparams.tensorboard_log_dir, hparams.models_dir,
                            checkpoint_name, hparams.patience, hparams.min_delta)
