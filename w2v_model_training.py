import os.path

import gensim
import pandas as pd
import spacy
import re

from config import parser


def train_model(model_path, processed_file_path, window_size):
    print("train_model: -------------------> ")
    try:
        print("train_model: initialize model")
        model = gensim.models.Word2Vec(corpus_file=processed_file_path, vector_size=300, window=window_size,
                                       min_count=1,
                                       workers=20)
        print("train_model: train model")
        model.train(corpus_file=processed_file_path, total_words=model.corpus_total_words, epochs=5)
        # Save the word vector of the model to directory
        print("train_model: save model to directory")
        word_vectors = model.wv
        word_vectors.save(model_path)

    except Exception as e:
        print("train_model: Exception: " + str(e))
        raise e

    print("train_model: <------------------- ")


def preprocess_data_for_w2v(df, processed_file_path, tokenizer, title_col):
    print("preprocess_data_for_w2v: -------------------> ")

    #
    df[title_col] = df[title_col].astype(str)
    # Clean text for the model training
    df['clean_sent'] = df[title_col].apply(lambda text: text.lower())
    # Remove non Ascii letters
    df['clean_sent'] = df['clean_sent'].apply(lambda text: "".join(c for c in text if ord(c) < 128))
    # Separate words
    df['clean_sent'] = df['clean_sent'].apply(lambda text: " ".join(a.text for a in tokenizer(text)))
    # Replace dot without space
    df['clean_sent'] = df['clean_sent'].apply(lambda text: text.replace(".", ""))
    # Clean text from special characters
    df['clean_sent'] = df['clean_sent'].apply(lambda text: re.sub('[^A-Za-z0-9 ]+', ' ', text.strip()))
    # Remove spaces
    df['clean_sent'] = df['clean_sent'].apply(lambda text: " ".join(text.split()))

    df['clean_sent'].to_csv(processed_file_path, sep='\n', index=False, header=False)

    print("preprocess_data_for_w2v: <------------------- ")


if __name__ == '__main__':
    hparams = parser.parse_args()
    spacy_tokenizer = spacy.blank('en')
    train = pd.read_csv(hparams.train_dataset_path,
                        keep_default_na=False,
                        na_values=['$$$__$$$'])

    preprocess_data_for_w2v(train, hparams.w2v_input_path, spacy_tokenizer, "title")
    print(f'train with window size - {hparams.w2v_window_size}')
    if not os.path.exists(hparams.w2v_model_dir):
        os.makedirs(hparams.w2v_model_dir)
    final_w2v_model_path = os.path.join(hparams.w2v_model_dir,
                                        f"{hparams.w2v_model_path_prefix}_window_{hparams.w2v_window_size}.kv")
    train_model(final_w2v_model_path, hparams.w2v_input_path, hparams.w2v_window_size)
