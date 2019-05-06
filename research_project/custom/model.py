import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from tqdm import trange
import logging


total_word_to_idx = {}
word_to_idx = {}
tag_to_idx = {}

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        # maps each token to an embedding_dim vector
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # the LSTM takens embedded sentence
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)

        # fc layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)
    
    def forward(self, s):
        # apply the embedding layer that maps each token to its embedding
        s = self.embedding(s)   # dim: batch_size x batch_max_len x embedding_dim
                    
        # run the LSTM along the sentences of length batch_max_len
        s, _ = self.lstm(s)     # dim: batch_size x batch_max_len x lstm_hidden_dim                
                    
        # reshape the Variable so that each row contains one token
        s = s.view(-1, s.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim
                    
        # apply the fully connected layer and obtain the output for each token
        s = self.fc(s)          # dim: batch_size*batch_max_len x num_tags
        
        return F.log_softmax(s, dim=1)   # dim: batch_size*batch_max_len x num_tags


class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def loss_function(outputs, labels):
    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)  
    
    # mask out 'PAD' tokens
    mask = (labels >= 0).float()
    
    # the number of tokens is the sum of elements in mask
    num_tokens = int(torch.sum(mask).data[0])
    
    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask
    
    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_tokens


def create_index_dicts():
    word_counter = {}
    current_index = 0
    with open('../nltk/train.txt') as f:
        for line in f:
            if line.isspace():
                continue
            else:
                word = line.split()[0]
                if word not in word_to_idx:
                    word_to_idx[word] = current_index
                    current_index = current_index + 1
                    word_counter[word] = 1
                else:
                    word_counter[word] = word_counter[word] + 1
    with open('../nltk/test.txt') as f:
        for line in f:
            if line.isspace():
                continue
            else:
                word = line.split()[0]
                if word not in word_to_idx:
                    word_to_idx[word] = current_index
                    current_index = current_index + 1
                    word_counter[word] = 1
                else:
                    word_counter[word] = word_counter[word] + 1
    with open('../nltk/dev.txt') as f:
        for line in f:
            if line.isspace():
                continue
            else:
                word = line.split()[0]
                if word not in word_to_idx:
                    word_to_idx[word] = current_index
                    current_index = current_index + 1
                    word_counter[word] = 1
                else:
                    word_counter[word] = word_counter[word] + 1
    total_word_to_idx = {word: idx for word, idx in word_to_idx.items() if word_counter[word] > 3}
    total_word_to_idx['<UNK>'] = current_index
    total_word_to_idx['<PAD>'] = current_index + 1
    tag_to_idx['B-PER'] = 0
    tag_to_idx['I-PER'] = 1
    tag_to_idx['B-ORG'] = 2
    tag_to_idx['I-ORG'] = 3
    tag_to_idx['B-LOC'] = 4
    tag_to_idx['I-LOC'] = 5
    tag_to_idx['B-MISC'] = 6
    tag_to_idx['I-MISC'] = 7
    tag_to_idx['O'] = 8

def get_tweets_as_idx(path):
    sentences = []
    tag_sentences = []
    with open(path) as f:    
        sentence = []
        tag_sentence = []
        for line in f:
            if line.isspace():
                if not sentence:
                    continue
                sentences.append(sentence)
                sentence = []
                tag_sentences.append(tag_sentence)
                tag_sentence = []
            else:
                tokens = line.split()
                word = tokens[0]
                tag = ''
                if len(tokens[2]) == 1:
                    tag = tokens[2]
                else:
                    tag = tokens[2][2:]
                if word not in total_word_to_idx:
                    sentence.append(total_word_to_idx['<UNK>'])
                else:
                    sentence.append(total_word_to_idx[word])
                tag_sentence.append(tag_to_idx[tag])

    return sentences, tag_sentences


def data_iterator(sents, tags, batch_size, shuffle=False):
    size = len(sents)
    pad_ind = total_word_to_idx['<PAD>']
    order = list(range(size))
    if shuffle:
        random.seed(230)
        random.shuffle(order)
    for i in range((size + 1) // batch_size):
        # grab chunk of sentences
        batch_sents = [sents[idx] for idx in order[i*batch_size:(i+1)*batch_size]]
        batch_tags = [tags[idx] for idx in order[i*batch_size:(i+1)*batch_size]]
        batch_max_len = max([len(s) for s in batch_sents])
        batch_data = pad_ind*np.ones((len(batch_sents), batch_max_len))
        batch_labels = -1*np.ones((len(batch_sents), batch_max_len))

        # fill numpy array
        for j in range(len(batch_sents)):
            cur_len = len(batch_sents[j])
            # fill row in numpy matrix with content of current sentence
            batch_data[j][:cur_len] = batch_sents[j]
            batch_labels[j][:cur_len] = batch_tags[j]
        
        batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
        batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

        yield batch_data, batch_labels


def train(model, optimizer, loss_fn, data_iterator, params, num_steps):
    model.train()
    summ = []
    loss_avg = RunningAverage()
    t = trange(num_steps)
    for i in t:
        train_batch, labels_batch = next(data_iterator)
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.update(loss.data[0])
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def accuracy(outputs, labels):
    labels = labels.ravel()
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(np.sum(mask))


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
    model.eval()
    summ = []
    for _ in range(num_steps):
        data_batch, labels_batch = next(data_iterator)
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)
        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()
        summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def predict():
    tests = get_tweets_as_idx('../nltk/train.txt')
    test_data_iterator = data_iterator(tests[0], tests[1], 16)
    



def train_and_eval(model, trains, vals, optimizer, loss_fn, metrics, params):
    train_data = trains[0]
    train_labels = trains[1]
    train_size = len(train_data)
    val_data = vals[0]
    val_labels = vals[1]
    val_size = len(val_data)
    best_val_acc = 0.0
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_iterator(train_data, train_labels, params.batch_size)
        train(model, optimizer, loss_fn, train_data_iterator, params, num_steps)

        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_iterator(val_data, val_labels, params.batch_size)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

                
def run():
    create_index_dicts()
    trains = get_tweets_as_idx('../nltk/train.txt')
    vals = get_tweets_as_idx('../nltk/dev.txt')

    vocab_size = len(total_word_to_idx)
    params = {'learning_rate': 1e-3, 'lstm_hidden_dim': 50, 'embedding_dim': 50, \
        'number_of_tags': 9, 'vocab_size': vocab_size, 'batch_size': 16}
    model = Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    metrics = {'accuracy': accuracy}
    train_and_eval(model, trains, vals, optimizer, loss_function, metrics, params)
