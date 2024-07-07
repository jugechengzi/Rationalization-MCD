




Due to different versions of torch, you may need to replace "cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels)" with "cls_loss = args.cls_lambda * F.cross_entropy(forward_logit, labels.long())"



# MCD
This repo contains Pytorch implementation of MCD (NeurIPS 2023 paper: [D-Separation for Causal Self-Explanation](https://arxiv.org/abs/2309.13391)).  Most of our code are built on top of our previous work FR.

You can also refer to our team's other complementary work in this series：[FR (NeurIPS2022)](https://arxiv.org/abs/2209.08285), [DR (KDD 2023)](https://dl.acm.org/doi/abs/10.1145/3580305.3599299), [MGR (ACL 2023)](https://arxiv.org/abs/2305.04492), [MCD (NeurIPS 2023)](https://arxiv.org/abs/2309.13391), [DAR (ICDE 2024)](https://arxiv.org/abs/2312.04103).

**If the code has any bugs, please open an issue. We will be grateful for your help.**


I am happy to say that a recent paper called [SSR](https://arxiv.org/abs/2403.07955) has taken our previous work [FR](https://github.com/jugechengzi/FR) as a backbone. Congratulations on the great work of Yue et.al. and thanks for their citation. I also find that several recent works, such as [GR](https://ojs.aaai.org/index.php/AAAI/article/download/29783/31352) and [YOFO](https://arxiv.org/abs/2311.02344), have designed their experiments based on our open source code. Congratulations to all of them and I am really happy to know that my work is helping others.



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
