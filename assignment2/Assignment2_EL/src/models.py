import random
from operator import attrgetter
import pandas as pd
from difflib import SequenceMatcher
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


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

    # features will be similarity between candidate mention and surface names, and probability
    # predict variable will be is the candidate the ground truth
    def fit(self, dataset):
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