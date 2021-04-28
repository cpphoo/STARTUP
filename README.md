# Self-training for Few-shot Transfer Across Extreme Task Differences (STARTUP)

## Introduction
This repo contains the official implementation of the following ICLR2021 paper:

**Title:** Self-training for Few-shot Transfer Across Extreme Task Differences  
**Authors:** Cheng Perng Phoo, Bharath Hariharan  
**Institution:** Cornell University  
**Arxiv:** https://arxiv.org/abs/2010.07734  
**Abstract:**  
Most few-shot learning techniques are pre-trained on a large, labeled "base dataset". In problem domains where such large labeled datasets are not available for pre-training (e.g., X-ray, satellite images), one must resort to pre-training in a different "source" problem domain (e.g., ImageNet), which can be very different from the desired target task. Traditional few-shot and transfer learning techniques fail in the presence of such extreme differences between the source and target tasks. In this paper, we present a simple and effective solution to tackle this extreme domain gap: self-training a source domain representation on unlabeled data from the target domain. We show that this improves one-shot performance on the target domain by 2.9 points on average on the challenging BSCD-FSL benchmark consisting of datasets from multiple domains.

### Requirements
This codebase is tested with:  
1. PyTorch 1.7.1
2. Torchvision 0.8.2
3. NumPy 
4. Pandas
5. wandb (used for logging. More here: https://wandb.ai/)




## Running Experiments 
### Step 0: Dataset Preparation
**MiniImageNet and CD-FSL:** Download the datasets for CD-FSL benchmark following step 1 and step 2 here: https://github.com/IBM/cdfsl-benchmark  
**tieredImageNet:** Prepare the tieredImageNet dataset following https://github.com/mileyan/simple_shot. Note after running the preparation script, you will need to split the saved images into 3 different folders: train, val, test. 

### Step 1: Teacher Training on the Base Dataset
We provide scripts to produce teachers for different base datasets. Regardless of the base datasets, please follow the following steps to produce the teachers:
1. Go into the directory `teacher_miniImageNet/` (`teacher_ImageNet/` for ImageNet)
2. Take care of the `TODO:` in  `run.sh` and `configs.py` (if applicable). 
3. Run `bash run.sh` to produce the teachers. 

Note that for miniImageNet and tieredImageNet, the training script is adapted based on the official script provided by the CD-FSL benchmark. For ImageNet, we simply download the pre-trained models from PyTorch and convert them to relevant format. 

### Step 2: Student Training
To train the STARTUP's representation, please follow the following steps:
1. Go into the directory `student_STARTUP/` (`student_STARTUP_no_self_supervision/` for the version without SimCLR)
2. Take care of the `TODO:` in  `run.sh` and `configs.py` 
3. Run `bash run.sh` to produce the student/STARTUP representation. 

### Step 3: Evaluation
To evaluate different representations, go into `evaluation/`, modify the  `TODO:` in  `run.sh` and `configs.py` and run `bash run.sh`. 


## Notes 
1. When producing the results for the submitted paper, we did not set `torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark` properly, thus causing non-deterministic behaviors. We have rerun our experiments and the updated numbers can be found here: https://docs.google.com/spreadsheets/d/1O1e9xdI1SxVvRWK9VVxcO8yefZhePAHGikypWfhRv8c/edit?usp=sharing. Although some of the numbers has changed, the conclusion in the paper remains unchanged. STARTUP is able to outperform all the baselines, bringing forth tremendous improvements to cross-domain few-shot learning. 
2. All the trainings are done on Nvidia Titan RTX GPU. Evaluation of different representations are performed using Nvidia RTX 2080Ti. Regardless of the GPU models, CUDA11 is used.  
3. This repo is built upon the official CD-FSL benchmark repo: https://github.com/IBM/cdfsl-benchmark/tree/9c6a42f4bb3d2638bb85d3e9df3d46e78107bc53. We thank the creators of the CD-FSL benchmark for releasing code to the public. 
4. If you find this codebase or STARTUP useful, please consider citing our paper: 
```
@inproceeding{phoo2021STARTUP,
    title={Self-training for Few-shot Transfer Across Extreme Task Differences},
    author={Phoo, Cheng Perng and Hariharan, Bharath},
    booktitle={Proceedings of the International Conference on Learning Representations},
    year={2021}
}
```
