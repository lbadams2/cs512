import models
from dataset import Dataset, Candidate
import pickle

if __name__ == '__main__':
    model = models.EmbedModel()
    trainset = Dataset.get('train')
    num_train_candidates = Candidate._count.__next__()
    model.fit(trainset, num_train_candidates)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))