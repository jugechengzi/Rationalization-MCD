

**update in 2024.0201: We apologize that the previous version had some minor bugs that didn't work straight away, we fixed them:**  
"from model import GenEncNoShareModel": We do not use GenEncNoShareModel. Just delete it.

if args.model_type!='sp':  
  for idx,p in model.layernorm2.named_parameters():  
    if p.requires_grad == True:  
      name3.append(idx)  
      p.requires_grad = False  
: This is in train_util.py. we also do not use it, just delete it.  




# MCD
This repo contains Pytorch implementation of MCD (NeurIPS 2023 paper: [D-Separation for Causal Self-Explanation](https://arxiv.org/abs/2309.13391)).  Most of our code are built on top of our previous work FR.

**If the code has any bugs, please open an issue. We will be grateful for your help.**



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
Preparing the code is really tedious, I will be appreciate if you star this repo before cloning it.



## Result
Please refer to [FR](https://github.com/jugechengzi/FR).  





## Acknowledgement

The code is largely based on [Car](https://github.com/code-terminator/classwise_rationale) and [DMR](https://github.com/kochsnow/distribution-matching-rationality). Most of the hyperparameters (e.g. the '--cls_lambda'=0.9) are also from them. We are grateful for their open source code.
