# A Hybrid Low-Rank Natural Gradient Descent Method (HyLo)

## Quick Start
0. **Clone this repository**
```
git clone https://github.com/hylo-dso/hylo.git
cd hylo
```

1. **Download singularity image from [Sylabs](https://cloud.sylabs.io/)** (Sylabs account required)
 * if you have not login to singularity on the system, generate a token on Sylabs, after running the command below, paste it to the command line interface
    ```
    singularity remote login
    ```
 * pull the singularity image from remote
    ```
    singularity pull --arch ppc64le library://hylo/dso/artifact:init
    ```

2. **Download datasets** (if datasets are not available locally)
  * [ImageNet-1k](https://image-net.org/download.php) (ImageNet account required)
  * [LGG MRI](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)


3. **Update config scripts** (i.e. ```run-*.sh``` files in this repository)
  * this is an optional step, for example, you might want to modify
    * path to singularity image/datasets,
    * hyperparameters,
    * distributed system configuration etc (if you are testing on a cluster other than [SciNet Mist](https://docs.scinet.utoronto.ca/index.php/Mist)).

4. **Start training with HyLo**, for example:
  * interactive mode:
    * CIFAR-10 + ResNet-32 (Classification)
      ```
      sh run-cifar10-resnet32.sh --world-size 32 --log-dir logs-cifar10-resnet32
      ```
    * ImageNet-1k + ResNet-50 (Classification)
      ```
      sh run-imagenet-resnet50.sh --world-size 64 --log-dir logs-imagenet-resnet50
      ```
    * Brain LGG
      ```
      sh run-brain-unet.sh --world-size 4 --log-dir logs-brain-unet
      ```
  * submit job to compute nodes via sbatch
    * CIFAR-10 + ResNet-32 (Classification)
      ```
      sbatch -p compute_full_node -N 8 -t 00:30:00 --gpus-per-node=4 --ntasks=1024 -J cf-rn32 -o cf-rn32.o%j --mail-type=ALL \
        run-cifar10-resnet32.sh --world-size 32 --log-dir logs-cifar10-resnet32
      ```
    * ImageNet-1k + ResNet-50 (Classification)
      ```
      sbatch -p compute_full_node -N 16 -t 2:00:00 --gpus-per-node=4 --ntasks=2048 -J img-rn50 -o img-rn50.o%j --mail-type=ALL \
        run-imagenet-resnet50.sh --world-size 64 --log-dir logs-imagenet-resnet50
      ```
    
    * Brain Segmentation + U-Net (Segmentation)
      ```
      sbatch -p compute_full_node -N 1 -t 00:30:00 --gpus-per-node=4 --ntasks=128 -J lgg-unet -o lgg-unet.o%j --mail-type=ALL \
        run-brain-unet.sh --world-size 4 --log-dir logs-brain-unet
      ```

## Requirements
```
pytorch==1.7.1
cudatoolkit==10.2.89
cudnn==7.6.5
torchinfo==1.6.5
tensorboard==2.4.1
torchvision==0.8.2
matplotlib==3.5.1
scikit-image==0.18.3
medpy==0.3.0
```
