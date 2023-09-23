# MCD
This repo contains Pytorch implementation of MCD (NeurIPS 2023 paper: [D-Separation for Causal Self-Explanation](https://github.com/jugechengzi/Rationalization-MCD/blob/main/arxiv.pdf)).  Most of our code are built on top of our previous work FR.
## Environments
torch 1.13.1+cu11.6.  
python 3.7.16.   
RTX3090  
## Datasets
Please refer to [FR](https://github.com/jugechengzi/FR).  

## Running example
### correlated Beer (Table 2)   

For the appearance aspect with sparsity being about 20%, run:   
python -u decouple_bcr.py --correlated 1 --data_type beer --lr 0.0001 --batch_size 128 --gpu 0 --sparsity_percentage 0.175 --epochs 150 --aspect 0

If you have any other questions, please send me an email. I am happy to provide further help if you star this repo.


## Result
Please refer to [FR](https://github.com/jugechengzi/FR).  





