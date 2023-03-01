from .dataset import collate_batch_amazon, get_data_loader_amazon, collate_batch_random_prefix_with_aspects_amazon
from functools import partial
from .utils import model_and_tokenizer_from_spec_cls, train_model, process_df, create_subsets_df
from .config import parser
import pandas as pd
import spacy
import pickle

if __name__ == '__main__':
    hparams = parser.parse_args()

    train = pd.read_csv(hparams.train_dataset_path,
                        keep_default_na=False, na_values=['_'])
    dev = pd.read_csv(hparams.dev_dataset_path,
                      keep_default_na=False, na_values=['_'])


    if hparams.mode in ['subsets', 'random','attributes']:
        spacy_tokenizer = spacy.blank("en")
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
    if hparams.mode!='attributes':
        collate_fn = partial(collate_batch_amazon, pad_token_id=tokenizer.pad_token_id)
        common_tags = []
    else:
        with open(hparams.common_tags_file,'rb') as f:
            common_tags = pickle.load(f)
            if bool(hparams.no_aspect_tokens):
                common_tags.append('UNKNOWN')
        collate_fn = partial(collate_batch_random_prefix_with_aspects_amazon, pad_token_id=tokenizer.pad_token_id,only_aspect_sentence=bool(hparams.only_aspect_sentence))
        tokenizer.add_special_tokens({'additional_special_tokens': [f'[{tag}]' for tag in common_tags]})
        model.bert.resize_token_embeddings(len(tokenizer))


    train_dl = get_data_loader_amazon(train, tokenizer, collate_fn,hparams.mode, bool(hparams.no_aspect_tokens),bool(hparams.only_aspect_sentence),common_tags,batch_size=hparams.batch_size, num_workers=4,
                                      shuffle=True)
    dev_dl = get_data_loader_amazon(dev, tokenizer, collate_fn,hparams.mode, bool(hparams.no_aspect_tokens),bool(hparams.only_aspect_sentence),common_tags,batch_size=hparams.batch_size, num_workers=4)

    # conduct training with given parameters
    train_model(model, train_dl, dev_dl, hparams.lr, hparams.batch_size, hparams.wd, hparams.num_epochs,
                hparams.warmpup_steps, hparams.seed, hparams.tensorboard_log_dir, hparams.models_dir,
                hparams.checkpoint_prefix, hparams.patience, hparams.min_delta)
