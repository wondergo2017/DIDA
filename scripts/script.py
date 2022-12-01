from DIDA.utils.mp import *
import os
import pandas as pd
import shutil
from pathlib import Path
import os.path as osp
import json
from collections import Counter
fdir=f"../logs/exp0"

resources=[0,1,2,3]
def func(dev,cfg):
    cmd,dir=cfg
    os.makedirs(f"{fdir}/{dir}",exist_ok=True)
    cmd += f' --device_id {dev}' + f' --log_dir "{fdir}/{dir}/" > "{fdir}/{dir}/logn.txt"'
    with open(f"{fdir}/{dir}/cmd.txt",'w') as file:
        file.write(cmd)
    print(cmd)
    os.system(cmd)

def get_configs():
    configs=[]
    for data in 'collab yelp synthetic-0.4 synthetic-0.6 synthetic-0.8'.split():
        for seed in range(3):
            configs.append([f'python main.py --dataset {data} --seed {seed} ',f'{data}-{seed}'])
    return configs

def run():
    configs=get_configs()
    mp_exec(resources,configs,func)

def show():
    pass
    res=[]
    for r,ds,fs in os.walk(fdir):
        for f in fs:
            if osp.splitext(f)[1]=='.json':
                info=json.load(open(osp.join(r,'info.json')))
                line={}
                line['dataset'] = dataset = info['dataset']
                if 'syn' in dataset:
                    line['w/o DS'] = info['train_auc']
                    line['w/ DS'] = info['test_auc']
                else:
                    line['w/o DS'] = info['test_auc']
                    line['w/ DS'] = info['test_test_auc']
                res.append(line)
                
    df = pd.DataFrame(res)
    df = df.groupby(by='dataset').agg(mean_WODS=('w/o DS','mean'),std_WODS=('w/o DS','std'),mean_WDS=('w/ DS','mean'),std_WDS=('w/ DS','std')).reset_index()
    df = df.applymap(lambda x:f"{x*100:.2f}" if isinstance(x,float) else x)
    df['w/o DS'] = df['mean_WODS']+ '+-' + df['std_WODS']
    df['w/ DS'] = df['mean_WDS']+ '+-' + df['std_WDS']
    df.drop(columns=['mean_WODS','std_WODS','mean_WDS','std_WDS'],inplace=True)
    
    print(df.to_string(),file=open(f"{fdir}/results.txt",'w'))

if __name__=='__main__':
    from argparse import ArgumentParser
    def get_args(args=None):
        parser=ArgumentParser()
        parser.add_argument('-t',type=str,default='show',choices=['show','run','debug'])
        args=parser.parse_args(args)
        return args

    args=get_args()
    t=args.t

    if t=='show':
        show()
    elif t=='run':
        run()