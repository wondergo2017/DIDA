# DIDA

## Dependencies

Require

- Python == 3.8
- PyTorch == 1.11
- PyTorch-Geometric == 2.0.3

Install other packages using following command at root directory of the repository

```
pip install -e .
```

## Dataset 

Download dataset at ./data from following links

```
Coming Soon.
```
## Usage

To run on one dataset, please execute following commands in the directory ./scripts

```
python main.py --dataset collab --log_dir ../logs/collab --device_id X 
python main.py --dataset yelp --log_dir ../logs/yelp --device_id X 
python main.py --dataset synthetic-0.6 --log_dir ../logs/synthetic-0.6 --device_id X 
```

To reproduce the main results, please execute following commands in the directory ./scripts. Change the 'resources' in the script.py as available GPU ids.

```
python script.py -t run
python script.py -t show
```

## Paper 

For more details, please see our paper Dynamic Graph Neural Networks Under Spatio-Temporal Distribution Shift which has been accepted at NeurIPS 2022. If this code is useful for your work, please consider to cite our paper.