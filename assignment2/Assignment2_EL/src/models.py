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