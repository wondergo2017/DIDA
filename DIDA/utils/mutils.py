import random 
import numpy as np
import torch
def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import shutil
from torch.utils.tensorboard import SummaryWriter
import json
import os.path as osp

def get_arg_dict(args):
    info_dict=args.__dict__
    ks=list(info_dict.keys())
    arg_dict={}
    for k in ks:
        v=info_dict[k]
        for t in [int,float,str,bool,torch.Tensor]:
            if isinstance(v,t):
                arg_dict[k]=v
                break
    return arg_dict


class EarlyStopping:
    """ EarlyStopping class to keep NN from overfitting. copied from nni
    """
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        """ EarlyStopping step on each epoch
        @params metrics: metric value
        @return : True if stop
        """

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False
        
    def reset(self):
        self.best=None

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

def is_empty_edges(edges):
    return edges.shape[1]==0

def map2id(l):
    '''encode list (unique) starting from 0.
    @return : mapping dict from l -> int.
    '''
    return dict(zip(l, range(len(l))))

def sorteddict(x, min=True,dim=1):
    """return dict sorted by values
    @params x: a dict
    @params min : whether from small to large. 
    """
    if min:
        return dict(sorted(x.items(), key=lambda item: item[dim]))
    else:
        return dict(sorted(x.items(), key=lambda item: item[dim])[::-1])


from gensim.models import Word2Vec
import gensim
def sen2vec(sentences, vector_size=32):
    """use gensim.word2vec to generate wordvecs, and average as sentence vectors. 
    if exception happens use zero embedding.
    @ params sentences : list of sentence
    @ params vector_size 
    @ return : sentence embedding 
    """
    sentences = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    print('word2vec done')
    embs = []
    for s in sentences:
        try:
            emb = model.wv[s]
            emb = np.mean(emb, axis=0)
        except Exception as e:
            print(e)
            emb = np.zeros(vector_size)
        embs.append(emb)
    embs = np.stack(embs)
    print(f'emb shape : {embs.shape}')
    return embs


from torch_geometric.utils.negative_sampling import negative_sampling
def hard_negative_sampling(edges,all_neg=False,inplace=False):
    ei = edges.numpy()

    # reorder
    nodes = list(set(ei.flatten()))
    nodes.sort()
    id2n = nodes
    n2id = dict(zip(nodes, np.arange(len(nodes))))

    ei_ = np.vectorize(lambda x: n2id[x])(ei)
    
    if all_neg:
        maxn = len(nodes)
        nei_ = []
        pos_e = set([tuple(x) for x in ei_.T])
        for i in range(maxn):
            for j in range(maxn):
                if i!=j and (i,j) not in pos_e:
                    nei_.append([i,j])
        nei_ = torch.LongTensor(nei_).T
    else:
        nei_ = negative_sampling(torch.LongTensor(ei_))
    nei = torch.LongTensor(np.vectorize(lambda x: id2n[x])(nei_.numpy()))
    return nei

def bi_negative_sampling(edges,num_nodes,shift):
    nes = edges.new_zeros((2,0))
    while nes.shape[1]<edges.shape[1]:
        num_need=edges.shape[1]-nes.shape[1]
        ne = negative_sampling(edges,num_nodes=num_nodes,num_neg_samples=4*num_need)
        mask = torch.logical_xor((ne[0]<shift), (ne[1]<shift))
        ne=ne[:,mask]
        nes=torch.concat([nes,ne[:,:num_need]],dim=-1)
    return nes

import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))

    