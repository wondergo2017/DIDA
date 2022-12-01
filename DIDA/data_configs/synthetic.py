import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data/Cross-Domain_data')
processed_datafile = f"{dataroot}/processed2"
synthetic_file=f'{processed_datafile}-sythetic2'

dataset='crossdomain'
testlength=5
vallength=1

P=0.6
SIGMA=0.05
TEST_P=0.1
TEST_SIGMA=0.0