# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import os
import random
import pandas as pd

# Set seed for reproducibility
torch.manual_seed(1)
random.seed(1)
torch.set_num_threads(8)  # Set the number of threads

# Define the base path for the project
base_path = "/home/jrzvnn/Documents/lfcm-racism/model"

# Define the LSTM Classifier model
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
        return log_probs

# Function to calculate accuracy
def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

# Function to load data from CSV file
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    text = df['tweet_text'].astype(str).tolist()
    label = df['label'].astype(int).tolist()
    return text, label

# Main training function
def train():
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 150
    EPOCH = 10
    BATCH_SIZE = 10

    # Load training and validation data
    train_text, train_label = load_data("/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/train.csv")
    valid_text, valid_label = load_data("/home/jrzvnn/Documents/Projects/lfcm-racism/Data/Processed_Data/valid.csv")

    # Define text and label fields for torchtext
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)

    # Build vocabulary using GloVe vectors
    text_field.build_vocab(train_text, vectors="glove.twitter.27B.100d")
    label_field.build_vocab(train_label)

    train_data = list(zip(train_text, train_label))
    valid_data = list(zip(valid_text, valid_label))

    train_examples = [data.Example.fromlist([text, label], [('tweet_text', text_field), ('label', label_field)]) for text, label in train_data]
    valid_examples = [data.Example.fromlist([text, label], [('tweet_text', text_field), ('label', label_field)]) for text, label in valid_data]

    train_dataset = data.Dataset(train_examples, [('tweet_text', text_field), ('label', label_field)])
    valid_dataset = data.Dataset(valid_examples, [('tweet_text', text_field), ('label', label_field)])

    # Create iterators for training and validation data
    train_iter, valid_iter = data.Iterator.splits(
        (train_dataset, valid_dataset),
        batch_sizes=(BATCH_SIZE, BATCH_SIZE),
        sort_key=lambda x: len(x.tweet_text),
        repeat=False
    )

    print("Len labels vocab: " + str(len(label_field.vocab)))
    print(label_field.vocab.itos)
    print("Used label len is: " + str(len(label_field.vocab) - 1))

    # Define class labels and weights for imbalance handling
    class_labels = ['NotRacist', 'Racist']
    class_weights = [1, 1]

    # Normalize class weights
    min_instances = min(class_weights)
    for i in range(0, len(class_weights)):
        class_weights[i] = 1 / (float(class_weights[i]) / min_instances)
    class_weights = torch.FloatTensor(class_weights)

    print("Class weights: ")
    print(class_weights)

    # Initialize the model, set word embeddings, loss function, and optimizer
    best_dev_acc = 0.0
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab), label_size=len(label_field.vocab) - 1,
                           batch_size=BATCH_SIZE)

    model.word_embeddings.weight.data = text_field.vocab.vectors
    loss_function = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    no_up = 0

    # Training loop
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i)
        print('now best dev acc:', best_dev_acc)
        dev_acc = evaluate_classes(class_labels, model, valid_iter, loss_function, 'dev')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('rm ' + base_path + '*.model')
            print('New Best Dev!!! ' + str(best_dev_acc))
            torch.save(model.state_dict(), base_path + 'best_model_acc_val.model')
            no_up = 0
        else:
            print('NOT improving Best Dev')
            no_up += 1
            if no_up >= 10:
                print('Ending because the DEV ACC does not improve')
                exit()

# Function to evaluate the model
def evaluate(model, eval_iter, loss_function, name='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for batch in eval_iter:
        sent, label = batch.tweet_text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data.item()

    avg_loss /= len(eval_iter)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g  acc:%g' % (avg_loss, acc))
    return acc

# Function to evaluate the model for each class
def evaluate_classes(class_labels, model, eval_iter, loss_function, name='dev'):
    model.eval()
    truth_res = []
    pred_res = []

    tp_classes = np.zeros(len(class_labels))
    fn_classes = np.zeros(len(class_labels))
    accuracies = np.zeros(len(class_labels))

    for batch in eval_iter:
        sent, label = batch.tweet_text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]

    for i, cur_pred_res in enumerate(pred_res):
        if cur_pred_res == truth_res[i]: tp_classes[truth_res[i]] += 1
        else: fn_classes[truth_res[i]] += 1

    for i in range(0, len(class_labels)):
        accuracies[i] = tp_classes[i] / float((tp_classes[i] + fn_classes[i]))

    acc = accuracies.mean()

    print(name + ' acc:%g' % (acc))
    return acc

# Function to train the model for one epoch
def train_epoch(model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []

    for batch in train_iter:
        sent, label = batch.tweet_text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data.item()
        count += 1
        if count % 1000 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count * model.batch_size, loss.data.item()))
        loss.backward()
        optimizer.step()

    avg_loss /= len(train_iter)
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g' % (i, avg_loss, get_accuracy(truth_res, pred_res)))

# Start training
train()
