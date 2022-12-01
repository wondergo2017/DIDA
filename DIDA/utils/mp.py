import torch.multiprocessing as mp
from queue import Queue
import time 
import random
import os

def mp_exec(resources,configs,func):
    '''
    @ resources : list of gpu devices
    @ configs : list of params
    @ func : f(dev,cfg)
    '''
    q=Queue()
    for res in resources:
        q.put(res)
    pool=mp.Pool()
    def put_back_dev(dev,cfg):
        def callback(*args):
            print(f"Device {dev} Finish cfg {cfg} ")
            q.put(dev)
            print(*args)
        return callback

    for idx,cfg in enumerate(configs):
        dev = q.get()
        print(f"Start config {cfg} on device {dev}")
        pool.apply_async(func,args=[dev,cfg],callback=put_back_dev(dev,cfg),error_callback=put_back_dev(dev,cfg))

    pool.close()
    pool.join()