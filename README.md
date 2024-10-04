# title_prefixes
Code for "Selling Flow Enhancement by Early Product Categorization" paper


# Data

The Amazon dataset used in paper's experiments is within the "data/" folder. The data is splitted into 3 files - train, dev and test sets. 

# Components

To run learning process execute the "train_procedure.py" file. All hyperparameters are controlled via "config.py" file. You can modify their values within the execution command via --\<parm name\> \<param value\> syntax.



# Implementation details

We implement our training framework with Pytorch. For the BERT-based models, 
we use the pretrained $BERT_{BASE}$ uncased language mode implemented with the HuggingFace library. Similarly to the work of Devlin et al., we use the AdamW optimizer with $\beta_1 {=} 0.9$, $\beta_2 {=} 0.999$, $\epsilon {=} 10^{-6}$ and weight decay of $10^{-2}$. In addition, we use a linear learning rate decay with $10^4$ warm-up steps. We tune our models with learning rates ${\in} \{10^{-5},3\cdot10^{-5},5\cdot10^{-5}\}$ over the validation set's accuracy metric. We use batch size of $64$ and we train our models for at most $10$ epochs with early stopping strategy of $\min\Delta {=} 0.05$ with patience of two epochs with respect to validation loss. 
For LSTM, we examine additional learning rates ${\in} \{10^{-2},10^{-3},10^{-4}\}$ and train W2V models (via the Gensim library) with a window size ${\in} \{4,6\}$ over the training set. 
Finally, we use Spacy's word tokenizer for title tokenization. 
