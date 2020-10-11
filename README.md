## DAN_PUlearning
This is the code on CIFAR-10 for DAN PU learning in the paper [Discriminative adversarial networks for positive-unlabeled learning](https://arxiv.org/abs/1906.00642v3).
* `func.py` contains the function log_sum_exp for computing in training process with gpu. 
* ` train.py` is the code of DAN on CIFAR-10. The default setting in this example is that positive labels are '0,1,8,9', and there are 3000 positive data which are randomly selected from the whole positive data. We consider here the single-training-set scenerio of PU learning, which means the setting here is 3000 labeled positive data and 50000 unlabeled data of CIFAR-10. So there is no need to know the number of P in U in this method.

## Prerequisites
`Python 3.6`
`Numpy` 
`sklearn`
`PyTorch`

##
Before running `train.py`, make sure that you set the parser argument `--gpu_id` to an idle gpu.



