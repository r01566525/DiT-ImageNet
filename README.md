# Project

Welcome! In this project, you'll implement a DiT-S model in JAX and train it on ImageNet.

## Guide 

### Compute Resources
- For Princeton students: we will primarily use the Adroit cluster for this project, please follow the guide below to start:
1. Request an account: complete the [Adroit registration form](https://forms.rc.princeton.edu/registration/?q=adroit).
2. Read the scheduler docs: review the [SLURM guide](https://researchcomputing.princeton.edu/support/knowledge-base/slurm#gpus) to understand GPU job submission.
3. We recommend using MIG A100 or V100 on Adroit for development and debugging.
- For external students: we will be using TPU from Google for this project.
1. Reach out to taiminglu@princeton.edu and provide your personal gmail address you'd like to use to access TPUs.
2. Please follow [our lab's TPU tutorial](https://github.com/TaiMingLu/TPU-Manual) to learn how to use TPUs.
3. Please use a single TPU v3-8 or TPU v3-16 for development and debugging. Do not access or use TPUs that aren't created by yourself.
4. Please only use the local storage on TPU VMs and do not store anything in buckets.

### JAX
This project uses [JAX](https://github.com/jax-ml/jax). This community [learning guide](https://github.com/rcrowe-google/Learning-JAX) may be helpful.

### Dataset
For this project, please use the [ImageNet-100 dataset](https://www.kaggle.com/datasets/ambityga/imagenet100) (a subset of ImageNet-1k with 100 randomly selected classes).

### DiT
* Please familiarize yourself with the background knowledge, including [Transformers](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com) and [Diffusion models](https://arxiv.org/abs/2006.11239).
* It's helpful to understand DiT models: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
* Before starting, we recommend reviewing [FID](https://arxiv.org/abs/1706.08500), the evaluation metric.

## Goal
* Your goal is to tune training hyperparameters and other training settings for DiT-S to obtain an FID-10K score lower than 20 at resolution 256×256. Below are some example images generated from a DiT-S model with FID=19.8.
<p align="center">
<img src="./images/resolution256_fid19.8.png" width=50% height=50% 
class="center">
</p>

* For a more advanced challenge: try tweaking the architectures to optimize FID, but stay within 33M parameters (the size of DiT-S).

## Instructions
* Invite the GitHub account @r01566525 to your working GitHub repo.
* Please work on this project independently.
* Feel free to refer to other resources or tutorials.
* Please implement the codebase yourself as much as possible, including the attention module and the training loop; you may use AI or refer to others' code in an assisting capacity only, and please be prepared to explain all the code.


## Weekly Reports
* Please submit a weekly report (within 3 pages each week, keep in the same Google Doc) at the end of each week.
* Feel free to structure it yourself: you can include progress, issues / solutions, results, plots, or any other relevant things. Feel free to report negative results too, i.e., what has been tried but didn't work.
* Please clearly indicate in which parts of the code you used others' code (link source) or AI, and in what capacity.


## Contact
Please send your weekly reports to r01566525@gmail.com. You can schedule an initial meeting with a graduate student by emailing this address; you are also welcome to contact the graduate students in our group through this email with any questions you may have.
