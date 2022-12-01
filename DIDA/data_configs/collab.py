import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data/Cross-Domain_data')
processed_datafile = f"{dataroot}/processed2"

dataset = 'crossdomain'
testlength = 5
vallength = 1
length = 16
split = 0
