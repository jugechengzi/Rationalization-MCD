# MCD
This repo contains Pytorch implementation of MCD (NeurIPS 2023 paper: D-Separation for Causal Self-Explanation).  Most of our code are built on top of our previous work FR.
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


## Result
Taking appearance aspect with sparsity being about 20% as an example:  
You will get a result like "best_dev_epoch=96" at last. Then you need to find the result corresponding to the epoch with number "96". The results are as follows: 

Train time for epoch #96 : 15.648836 second  
traning epoch:96 recall:0.7977 precision:0.8584 f1-score:0.8269 accuracy:0.8331  
Validate  
dev epoch:96 recall:0.7382 precision:0.8115 f1-score:0.7731 accuracy:0.7834  
Validate Sentence
dev dataset : recall:0.7883 precision:0.7919 f1-score:0.7901 accuracy:0.7906  
Annotation  
annotation dataset : recall:0.8321 precision:1.0000 f1-score:0.9083 accuracy:0.8344  
The annotation performance: sparsity: 19.9373, precision: 80.0280, recall: 86.1792, f1: 82.9898  
Annotation Sentence  
annotation dataset : recall:0.8635 precision:0.9987 f1-score:0.9262 accuracy:0.8643  
Rationale  
rationale dataset : recall:0.7692 precision:0.9986 f1-score:0.8690 accuracy:0.7714  

The line "The annotation performance: sparsity: 19.9373, precision: 80.0280, recall: 86.1792, f1: 82.9898 " corresponds to the results reported in Table 2. These are not exactly the same as the results in Table 2, because we report in Table 2 the average results of five experiments.





