import models
from dataset import Dataset, Candidate
import pickle
import torch

def run_step(step_num):
    if step_num == 0:
        model = models.RandomModel()
    elif step_num == 1:
        model = models.PriorModel()
    elif step_num == 2:
        model = models.SupModel()
    elif step_num == 3:
        print('Training will take 6-7 minutes')
        model = models.EmbedModel()
    elif step_num == 4:
        print('Training will take 30-40 minutes')
        gru = models.GRU()
        model = models.NeuralModel(gru)
    else:
        raise ValueError('Invalid step number')
    trainset = Dataset.get('train')
    num_train_candidates = Candidate.get_count()
    model.fit(trainset, num_train_candidates)
    print('Training finished!')

    for dsname in Dataset.ds2path.keys():
        ds = Dataset.get(dsname)
        pred_cids = model.predict(ds)
        print(dsname, ds.eval(pred_cids))

def eval_pre_trained():
    model = models.GRU()
    model.load_state_dict(torch.load('../../assign2.model'))
    model.eval()
    net = models.NeuralModel(model)
    ds = Dataset.get('ace')
    pred_cids = net.predict(ds)
    print('ace', ds.eval(pred_cids))

if __name__ == '__main__':
    run_step(0)
    #eval_pre_trained()