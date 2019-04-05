import models
from dataset import Dataset, Candidate
import pickle

def run_original_model():
    model = models.EmbedModel()
    trainset = Dataset.get('train')
    num_train_candidates = Candidate._count.__next__()
    model.fit(trainset, num_train_candidates)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))
    

def run_neural_model():
    gru = models.GRU()
    model = models.NeuralModel(gru)
    trainset = Dataset.get('train')
    num_train_candidates = Candidate._count.__next__()
    model.fit(trainset, num_train_candidates)
    print('Training finished!')


if __name__ == '__main__':
    run_neural_model()