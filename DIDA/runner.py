import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
from DIDA.utils.mutils import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from DIDA.utils.inits import prepare
from DIDA.utils.loss import EnvLoss
from DIDA.utils.util import init_logger, logger

from torch_geometric.utils import negative_sampling
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
class Runner(object):

    def __init__(self, args, model, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.model = model
        self.writer = writer
        self.len = len(data['train']['edge_index_list'])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        x = data['x'].to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        self.loss = EnvLoss(args)
        print('total length: {}, test length: {}'.format(
            self.len, args.testlength))

    def train(self, epoch, data):
        args = self.args
        self.model.train()
        optimizer = self.optimizer
        conf_opt = self.conf_opt

        embeddings, cs, ss = self.model([
            data['edge_index_list'][ix].long().to(args.device)
            for ix in range(self.len)
        ], self.x)
        device = cs[0].device
        ss = [s.detach() for s in ss]

        # test
        val_auc_list = []
        test_auc_list = []
        train_auc_list = []
        for t in range(self.len - 1):
            z = cs[t]
            _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            auc, ap = self.loss.predict(z, pos_edge, neg_edge,
                                        self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        # train
        edge_index = []
        edge_label = []
        epoch_losses = []
        tsize = []
        for t in range(self.len_train - 1):
            z = embeddings[t]
            pos_edge_index = prepare(data, t + 1)[0]
            if args.dataset == 'yelp':
                neg_edge_index = bi_negative_sampling(pos_edge_index,
                                                      args.num_nodes,
                                                      args.shift)
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) *
                    args.sampling_times)
            edge_index.append(
                torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(self.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        cy = cal_y(cs, self.model.cs_decoder)
        sy = cal_y(ss, self.model.ss_decoder)

        conf_loss = cal_loss(sy, edge_label)
        causal_loss = cal_loss(cy, edge_label)

        env_loss = torch.tensor([]).to(device)
        intervention_times = args.n_intervene
        la = args.la_intervene

        if epoch < args.warm_epoch:
            la = 0

        if intervention_times > 0 and la > 0:
            if args.intervention_mechanism == 0:
                # slower version of spatial-temporal
                for i in range(intervention_times):
                    s1 = np.random.randint(len(sy))
                    s = torch.sigmoid(sy[s1]).detach()
                    conf = s * cy
                    # conf=self.model.comb_pred(cs,)
                    env_loss = torch.cat(
                        [env_loss,
                         cal_loss(conf, edge_label).unsqueeze(0)])
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 1:
                # only spatial
                sy = torch.sigmoid(sy).detach().split(tsize)
                cy = cy.split(tsize)
                for i in range(intervention_times):
                    conf = []
                    for j, t in enumerate(tsize):
                        s1 = np.random.randint(len(sy[j]))
                        s1 = sy[j][s1]
                        conf.append(cy[j] * s1)
                    conf = torch.cat(conf, dim=0)
                    env_loss = torch.cat(
                        [env_loss,
                         cal_loss(conf, edge_label).unsqueeze(0)])
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 2:
                # only temporal 
                alle = torch.cat(edge_index, dim=-1)
                v, idxs = torch.sort(alle[0])
                c = v.bincount()
                tsize = c[c.nonzero()].flatten().tolist()

                sy = torch.sigmoid(sy[idxs]).detach().split(tsize)
                cy = cy[idxs].split(tsize)
                edge_label = edge_label[idxs].split(tsize)

                crit = torch.nn.BCELoss(reduction='none')
                elosses = []
                for j, t in tqdm(enumerate(tsize)):
                    s1 = torch.randint(len(sy[j]),
                                       (intervention_times, 1)).flatten()
                    alls = sy[j][s1].unsqueeze(-1)
                    allc = cy[j].expand(intervention_times, cy[j].shape[0])
                    conf = allc * alls
                    alle = edge_label[j].expand(intervention_times,
                                                edge_label[j].shape[0])
                    env_loss = crit(conf.flatten(), alle.flatten()).view(
                        intervention_times, sy[j].shape[0])
                    elosses.append(env_loss)
                env_loss = torch.cat(elosses, dim=-1).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 3:
                # faster approximate version of spatial-temporal
                select=torch.randperm(len(sy))[:intervention_times].to(sy.device)
                alls=torch.sigmoid(sy).detach()[select].unsqueeze(-1) # [I,1]
                allc=cy.expand(intervention_times,cy.shape[0]) # [I,E]
                conf=allc*alls
                alle=edge_label.expand(intervention_times,edge_label.shape[0])
                crit=torch.nn.BCELoss(reduction='none')
                env_loss=crit(conf.flatten(),alle.flatten())
                env_loss=env_loss.view(intervention_times,sy.shape[0]).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss*intervention_times)
                penalty = env_mean+env_var
            else:
                raise NotImplementedError('intervention type not implemented')
        else:
            penalty = 0

        loss = causal_loss + la * penalty

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, train_auc_list, val_auc_list, test_auc_list

    def run(self):
        args = self.args
        
        minloss = 10
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.optimizer = optim.Adam(
            [p for n, p in self.model.named_parameters() if 'ss' not in n],
            lr=args.lr,
            weight_decay=args.weight_decay)
        if args.learns:
            self.conf_opt = optim.Adam(
                [p for n, p in self.model.named_parameters() if 'ss' in n],
                lr=args.lr,
                weight_decay=args.weight_decay)

        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                epoch_losses, train_auc_list, val_auc_list, test_auc_list = self.train(
                    epoch, self.data['train'])
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)

                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)

                # update the best results.
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    test_results = self.test(epoch, self.data['test'])

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(',')
                    measure_dict = dict(
                        zip(metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results))

                    patience = 0
                    
                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss,time.time() - t0))
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}"
                    )

                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc
                   ] + test_results + [epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(',')
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics],columns=metrics_des)
        print(df)
        return metrics_dict

    def test(self, epoch, data):
        args = self.args

        train_auc_list = []

        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings, cs, ss = self.model([
            data['edge_index_list'][ix].long().to(args.device)
            for ix in range(self.len)
        ], self.x)

        for t in range(self.len - 1):
            z = cs[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge,
                                        self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list)
        ]
