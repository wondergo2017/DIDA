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
https://drive.google.com/file/d/19SOqzYEKvkna6DKd74gcJ50Wd4phOHr3/view?usp=share_link
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

For more details, please see our paper [Dynamic Graph Neural Networks Under Spatio-Temporal Distribution Shift](https://openreview.net/pdf?id=1tIUqrUuJxx) which has been accepted at NeurIPS 2022. If this code is useful for your work, please consider to cite our paper.

```
@inproceedings{zhang2022dynamic,
  title={Dynamic graph neural networks under spatio-temporal distribution shift},
  author={Zhang, Zeyang and Wang, Xin and Zhang, Ziwei and Li, Haoyang and Qin, Zhou and Zhu, Wenwu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
