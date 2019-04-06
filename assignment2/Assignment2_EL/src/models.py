import random
from operator import attrgetter
import pandas as pd
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from numpy.linalg import norm
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RandomModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        # fill this function if your model requires training
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(random.choice(mention.candidates).id if mention.candidates else 'NIL')
        return pred_cids

class PriorModel:
    def __init__(self):
        pass

    def fit(self, dataset):
        # fill this function if your model requires training
        pass

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            pred_cids.append(max(mention.candidates, key=attrgetter('prob')).id if mention.candidates else 'NIL')
        return pred_cids

class SupModel:
    def __init__(self):
        self.logreg = LogisticRegression()

    def fit(self, dataset, candidate_count):
        df = self.get_dataframe(dataset)
        x_cols = ['text_sim', 'prob']
        x = df[x_cols]
        y = df['y']
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        self.logreg.fit(x, y)
        #y_pred = logreg.predict(x_test)
        #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

    def predict(self, dataset):
        pred_cids = []
        for mention in dataset.mentions:
            if mention.candidates:
                df = self.get_mention_dataframe(mention)
                conf_probs = self.logreg.predict_proba(df)[:,1]
                index_max = max(range(len(conf_probs)), key=conf_probs.__getitem__)
                pred_cids.append(mention.candidates[index_max].id)
            else:
                pred_cids.append('NIL')
        return pred_cids

    def get_mention_dataframe(self, mention):
        text_sim = []
        prior_prob = []
        surface_name = mention.surface
        for candidate in mention.candidates:
            sim = SequenceMatcher(None, candidate.name, surface_name).ratio()
            prob = candidate.prob
            text_sim.append(sim)
            prior_prob.append(prob)
        df = pd.DataFrame(list(zip(text_sim, prior_prob)), columns=['text_sim', 'prob'])
        return df

    def get_dataframe(self, dataset):
        text_sim = []
        prior_prob = []
        y = []
        for mention in dataset.mentions:
            surface_name = mention.surface
            ground_truth_id = mention.gt.id
            for candidate in mention.candidates:
                sim = SequenceMatcher(None, candidate.name, surface_name).ratio()
                prob = candidate.prob
                is_gt = ground_truth_id == candidate.id
                text_sim.append(sim)
                prior_prob.append(prob)
                y.append(is_gt)
        df = pd.DataFrame(list(zip(text_sim, prior_prob, y)), columns=['text_sim', 'prob', 'y'])
        return df

class EmbedModel:
    def __init__(self):
        self.logreg = LogisticRegression()

    def fit(self, dataset, candidate_count):
        feature_matrix, target_matrix = self.get_training_dataframe(dataset, candidate_count)
        self.logreg.fit(feature_matrix, target_matrix.ravel())

    def predict(self, dataset):
        pred_cids = []
        with open("../data/embeddings/ent2embed.pk", "rb") as ent_embed_file:
            ent2embed = pickle.load(ent_embed_file)
            with open("../data/embeddings/word2embed.pk", "rb") as word_embed_file:
                word2embed = pickle.load(word_embed_file)
                for mention in dataset.mentions:
                    if mention.candidates:
                        surface_name = mention.surface
                        current_candidate = 0
                        candidate_count = len(mention.candidates)
                        feature_matrix = np.empty(shape=(candidate_count, 603))
                        #print('Getting matrix for  ' + mention.surface)
                        for candidate in mention.candidates:                            
                            sim = SequenceMatcher(None, candidate.name, surface_name).ratio()
                            prob = candidate.prob
                            candidate_name = candidate.name.replace(' ', '_')
                            cand_embed = np.array(ent2embed[candidate_name])
                            context_embed_sum = np.zeros(shape=300)
                            total_context = mention.contexts[0] + mention.contexts[1]
                            for word in total_context:
                                if word not in word2embed:
                                    continue
                                word_embed = np.array(word2embed[word])
                                context_embed_sum = context_embed_sum + word_embed
                            cos_sim = np.dot(cand_embed, context_embed_sum)/norm(cand_embed)*norm(context_embed_sum)
                            hc_feature_array = np.array([sim, prob, cos_sim])
                            row = np.concatenate([cand_embed, context_embed_sum, hc_feature_array])
                            feature_matrix[current_candidate] = row
                            current_candidate = current_candidate + 1
                        
                        conf_probs = self.logreg.predict_proba(feature_matrix)[:,1]                        
                        index_max = max(range(len(conf_probs)), key=conf_probs.__getitem__)
                        pred_cids.append(mention.candidates[index_max].id)
                        #print('logreg predict done for ' + mention.surface)
                    else:
                        pred_cids.append('NIL')

        return pred_cids

    def get_training_dataframe(self, dataset, candidate_count):
        feature_matrix = np.empty(shape=(candidate_count, 603))
        target_matrix = np.empty(shape=(candidate_count, 1))
        current_candidate = 0
        with open("../data/embeddings/ent2embed.pk", "rb") as ent_embed_file:
            ent2embed = pickle.load(ent_embed_file)
            with open("../data/embeddings/word2embed.pk", "rb") as word_embed_file:
                word2embed = pickle.load(word_embed_file)
                for mention in dataset.mentions:
                    surface_name = mention.surface
                    ground_truth_id = mention.gt.id
                    for candidate in mention.candidates:
                        sim = SequenceMatcher(None, candidate.name, surface_name).ratio()
                        prob = candidate.prob
                        is_gt = ground_truth_id == candidate.id
                        target_matrix[current_candidate] = is_gt
                        candidate_name = candidate.name.replace(' ', '_')
                        cand_embed = np.array(ent2embed[candidate_name])
                        context_embed_sum = np.zeros(shape=300)
                        total_context = mention.contexts[0] + mention.contexts[1]
                        for word in total_context:
                            if word not in word2embed:
                                continue
                            word_embed = np.array(word2embed[word])
                            context_embed_sum = context_embed_sum + word_embed
                        cos_sim = np.dot(cand_embed, context_embed_sum)/norm(cand_embed)*norm(context_embed_sum)
                        hc_feature_array = np.array([sim, prob, cos_sim])
                        row = np.concatenate([cand_embed, context_embed_sum, hc_feature_array])
                        feature_matrix[current_candidate] = row
                        current_candidate = current_candidate + 1
                                            
        return feature_matrix, target_matrix

class GRU(nn.Module):
    INPUT_SIZE = 600
    HIDDEN_SIZE = 100
    OUTPUT_SIZE = 1

    def __init__(self):
        super(GRU, self).__init__()
        #self.hidden_size = 10
        self.gru = nn.GRU(self.INPUT_SIZE, self.HIDDEN_SIZE)
        self.linear = nn.Linear(self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        #self.sm = nn.Softmax(self.OUTPUT_SIZE)
        # make rows sum to 1, dim=0 would make columns sum to 1
        #self.sm = nn.Softmax(dim=1)
        self.sm = nn.Sigmoid()
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, input, hidden):
        _, hn = self.gru(input, hidden)
        # reduce from 3 to 2 dimensions
        rearranged = hn.view(hn.size()[1], hn.size(2))
        out1 = self.linear(rearranged)
        out2 = self.sm(out1)
        return out2

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.HIDDEN_SIZE))


class CandidateDataset(Dataset):
    def __init__(self, x, y):
        self.len = x.shape[0]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.x_data = torch.as_tensor(x, device=device, dtype=torch.float)
        self.y_data = torch.as_tensor(y, device=device, dtype=torch.float)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class NeuralModel():

    N_EPOCHS = 5
    BATCH_SIZE = 50
    
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.loss_f = nn.BCELoss()

    def fit(self, dataset, candidate_count):
        #candidate_ds = self.get_test_ds(candidate_count)
        candidate_ds = self.get_cand_ds(dataset, candidate_count)
        train_loader = DataLoader(dataset = candidate_ds, batch_size = self.BATCH_SIZE, shuffle = True)
        self.model.train()
        for epoch in range(self.N_EPOCHS):
            #print('starting epoch ' + str(epoch))
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                #print('starting batch ' + str(batch_idx) + ' epoch ' + str(epoch))
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                inputs = inputs.view(-1, inputs.size()[0], 600)
                # init hidden with number of rows in input
                y_pred = self.model(inputs, self.model.initHidden(inputs.size()[1]))
                #loss = self.loss_f(y_pred, torch.max(labels, 1)[1])
                #labels.squeeze_()
                # labels should be tensor with batch_size rows. Column the index of the class (0 or 1)
                loss = self.loss_f(y_pred, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if batch_idx % 500 == 499:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 500))
                    running_loss = 0.0
                #print('done batch ' + str(batch_idx) + ' epoch ' + str(epoch))
            #print('done epoch ' + str(epoch))

    def predict(self, dataset):
        pred_cids = []
        self.model.eval()
        with open("../data/embeddings/ent2embed.pk", "rb") as ent_embed_file:
            ent2embed = pickle.load(ent_embed_file)
            with open("../data/embeddings/word2embed.pk", "rb") as word_embed_file:
                word2embed = pickle.load(word_embed_file)
                for mention in dataset.mentions:
                    if mention.candidates:
                        current_candidate = 0
                        candidate_count = len(mention.candidates)
                        feature_matrix = np.empty(shape=(candidate_count, 600))
                        #print('Getting matrix for  ' + mention.surface)
                        for candidate in mention.candidates:                            
                            candidate_name = candidate.name.replace(' ', '_')
                            cand_embed = np.array(ent2embed[candidate_name])
                            context_embed_sum = np.zeros(shape=300)
                            total_context = mention.contexts[0] + mention.contexts[1]
                            for word in total_context:
                                if word not in word2embed:
                                    continue
                                word_embed = np.array(word2embed[word])
                                context_embed_sum = context_embed_sum + word_embed
                            row = np.concatenate([cand_embed, context_embed_sum])
                            feature_matrix[current_candidate] = row
                            current_candidate = current_candidate + 1
                        
                        x_data = torch.as_tensor(feature_matrix, device='cpu', dtype=torch.float)
                        x_data = x_data.view(-1, x_data.size()[0], 600)                        
                        y_pred = self.model(x_data, self.model.initHidden(x_data.size()[1]))
                        # this gets max class and its energy for each candidate
                        max_cand_class = torch.max(y_pred, 1)
                        max_energy = 0
                        max_index = -1
                        for idx, class_label in enumerate(max_cand_class[1]):
                            class_label = class_label.item()
                            max_e = max_cand_class[0][idx].item()
                            if max_e > max_energy:
                                max_index = idx
                                max_energy = max_e
                        # this gets the index for the single candidate with max energy 
                        values, indices = torch.max(max_cand_class[0], 0)
                        # this gets the class (0 or 1) for the candidate with max energy
                        max_energy_class = max_cand_class[1][max_index]
                        #print('max index: ' + str(max_index) + ' max class ' + str(max_energy_class))
                        pred_cids.append(mention.candidates[max_index].id)
                    else:
                        pred_cids.append('NIL')

        return pred_cids

    def test_predict(self):
        feature_matrix = np.random.rand(10, 600)
        x_data = torch.as_tensor(feature_matrix, device='cpu', dtype=torch.float)
        x_data = x_data.view(1, x_data.size()[0], 600)
        y_pred = self.model(x_data, self.model.initHidden(x_data.size()[1]))
        # this gets max class and its energy for each candidate
        max_cand_class = torch.max(y_pred, 1)
        max_energy = 0
        max_index = -1
        for idx, class_label in enumerate(max_cand_class[1]):
            class_label = class_label.item()
            max_e = max_cand_class[0][idx].item()
            if class_label == 1 and max_e > max_energy:
                max_index = idx
                max_energy = max_e
        # this gets the index for the single candidate with max energy 
        values, indices = torch.max(max_cand_class[0], 0)
        # this gets the class (0 or 1) for the candidate with max energy
        max_energy_class = max_cand_class[1][max_index]
        print('')

    def get_test_ds(self, candidate_count):
        feature_matrix = np.random.rand(candidate_count, 600)
        target_matrix = np.zeros((candidate_count, 1), dtype=int)
        for i in range(candidate_count):
            if i % 5 == 0:
                target_matrix[i] = 1
        candidate_ds = CandidateDataset(feature_matrix, target_matrix)
        return candidate_ds

    # assignment2/Assignment2_EL
    def get_cand_ds(self, dataset, candidate_count):
        feature_matrix = np.empty(shape=(candidate_count, 600))
        target_matrix = np.empty(shape=(candidate_count, 1))
        current_candidate = 0
        with open("../data/embeddings/ent2embed.pk", "rb") as ent_embed_file:
            ent2embed = pickle.load(ent_embed_file)
            with open("../data/embeddings/word2embed.pk", "rb") as word_embed_file:
                word2embed = pickle.load(word_embed_file)
                for mention in dataset.mentions:
                    ground_truth_id = mention.gt.id
                    for candidate in mention.candidates:
                        is_gt = ground_truth_id == candidate.id
                        '''
                        target_row = np.zeros(shape=(1,2))
                        if is_gt:
                            target_row[0][1] = 1
                        else:
                            target_row[0][0] = 1
                        target_matrix[current_candidate] = target_row
                        '''
                        target_matrix[current_candidate] = is_gt
                        candidate_name = candidate.name.replace(' ', '_')
                        cand_embed = np.array(ent2embed[candidate_name])
                        context_embed_sum = np.zeros(shape=300)
                        total_context = mention.contexts[0] + mention.contexts[1]
                        for word in total_context:
                            if word not in word2embed:
                                continue
                            word_embed = np.array(word2embed[word])
                            context_embed_sum = context_embed_sum + word_embed
                        row = np.concatenate([cand_embed, context_embed_sum])
                        feature_matrix[current_candidate] = row
                        current_candidate = current_candidate + 1

        candidate_ds = CandidateDataset(feature_matrix, target_matrix)
        return candidate_ds