# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import os
import random
import numpy as np
import pandas as pd

torch.set_num_threads(4)
torch.manual_seed(1)
random.seed(1)

target = 'img_txt'
split_name = 'tweets.' + target
model_name = 'tweet_text'
out_file_name = 'tweet_embeddings/lstm_embeddings_classification/' + target
split_folder = ''

class_labels = ['Racist', 'NotRacist']

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        )

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs, self.hidden

def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    text = df['tweet_text'].astype(str).tolist()
    label = df['label'].astype(int).tolist()
    ids = df['entry_id'].astype(int).tolist()
    return text, label, ids

def dataset_loader(text_field, label_field, id_field, batch_size, split_name):
    text, label, ids = load_data("/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/comments1.csv")

    text_field.build_vocab(text, vectors="glove.twitter.27B.100d")
    label_field.build_vocab(label)
    id_field.build_vocab(ids)

    data_examples = [data.Example.fromlist([text_i, label_i, id_i], [('tweet_text', text_field), ('label', label_field), ('id', id_field)])
                     for text_i, label_i, id_i in zip(text, label, ids)]

    dataset = data.Dataset(data_examples, [('tweet_text', text_field), ('label', label_field), ('id', id_field)])

    return data.Iterator(dataset, batch_size=batch_size, sort_key=lambda x: len(x.tweet_text), repeat=False)

def get_embeddings(model, split_iter):
    embeddings = []

    model.eval()
    for batch in split_iter:
        sent, _, _ = batch.tweet_text, batch.label, batch.id
        cur_batch_size = sent.size()[1]
        model.batch_size = cur_batch_size
        model.hidden = model.init_hidden()
        _, hidden = model(sent.cpu())
        embeddings.append(hidden[0].view(-1).detach().numpy())

    return np.concatenate(embeddings)

def test():
    model_path = '/home/jrzvnn/Documents/Projects/lfcm-racism/Code/Model/LSTM_model/comment_text.model'
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    BATCH_SIZE = 8

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    id_field = data.Field(sequential=False)

    split_iter = dataset_loader(text_field, label_field, id_field, BATCH_SIZE, split_name)

    print("Len labels vocab: " + str(len(label_field.vocab)))
    print(label_field.vocab.itos)
    print("Used label len is: " + str(len(label_field.vocab) - 1))

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab), label_size=len(class_labels),
                           batch_size=BATCH_SIZE)

    model.word_embeddings.weight.data = text_field.vocab.vectors
    model.load_state_dict(torch.load(model_path))
    model = model.cpu()

    print(len(split_iter))

    print("Computing ...")
    embeddings = get_embeddings(model, split_iter)
    print(embeddings)

test()
print("DONE")