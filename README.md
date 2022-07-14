# A Hybrid Low-Rank Natural Gradient Descent Method (HyLo)

## Get Started

0. **Clone this repository**
```
git clone https://github.com/hylo-dso/hylo.git
cd hylo
```

1. **Download datasets** (if not available locally)
	* Please follow the instructions from the official websites listed below to download the datasets.
		* [ImageNet-1k](https://image-net.org/download.php) (ImageNet account required)
		* [LGG MRI](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
		* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Optional)
			* note: this step is optional for CIFAR-10, the dataset can be automatically downloaded by the training script. In this case, the CIFAR-10 dataset will be stored in the path specified by ```--data-dir``` in the ```main*.py``` scripts
	* note: the provided scripts assume that the datasets are stored under the directory ```$SCRATCH```

2. **Set up the environment**
	* **Option 1**: use the singularity image (for ppc64le architecture) with all the requirements. To download the singularity image from [Sylabs](https://cloud.sylabs.io/):
		* if you have not login to singularity on the cluster, generate a token on Sylabs, after running the command below, paste it to the command line interface
		    ```
		    singularity remote login
		    ```
		* pull the singularity image from remote
		    ```
		    singularity pull --arch ppc64le library://hylo/dso/artifact:init
		    ```
		* note: the provided scripts assume that this repository ```hylo``` is cloned into ```$SCRATCH``` and the singularity image (by default, named ```artifact_init.sif```) is downloaded into the directory ```hylo```.
	* **Option 2**: a list of required packages
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
	* note: the provided scripts assume that the singularity image is used to set up the environment. Other options will need some additional modifications in the scripts, which will be discussed in the next section.

3. **Update the scripts**
	* The ```.sh``` scripts consists of 4 main parts:
		1. environment set up (see the PRELOAD command),
		2. training configurations set up, e.g. model, dataset, hyperparameters (see the CMD command),
		3. distributed system configurations, e.g. the list of allocated nodes, master node, number of nodes
			* note: the scripts use TCP initialization. The url format is ```tcp://$MASTER_ADDR:$MASTER_PORT```, where the ```MASTER_PORT``` is set to 1234 by default.
		4. launch the processes on each node with ```torch.distributed.launch``` (see the launcher command)
	* Things might need to change:
		* If you are NOT using the singularity image, please update the PRELOAD command to set up the environment accordingly. 
			* For example, if you choose to install all requirements listed above in a conda environment, say, named env, then the PRELOAD command should be (or similar to):
				```
				PRELOAD="module load anaconda3;"
				PRELOAD+="source activate env;"
				```
		* On different number of GPUs, the hyperparameters are different. For example, the frequency (```--freq```) is scaled inversely with the number of GPUs. The learning rate, damping, target damping, weight decay might need to be adjusted.
		* The provided scripts for ResNet-32 + CIFAR-10 assume that the number of GPUs per node is 4 (```nproc_per_node=4```). Other scripts assume that ```nproc_per_node=4```. If the system you are using has a different setting, please update ```nproc_per_node``` in the CMD and LAUNCHER command accordingly.


## End-to-End Training

The commands run end-to-end training of the model + dataset. The accuracy and wall-clock time are written to a ```.csv``` file in the ```hylo``` directory. The checkpoint files (```.pth.tar```) are stored in the directory specified by the ```--log-dir``` arg of the ```main-*.py``` scripts.

* Multi-GPUs
	* (Image classification) ResNet-50 + ImageNet-1k
		```
		sh scripts/train-resnet32-cifar10-end-to-end.sh
		```
	* (Image classification) ResNet-32 + CIFAR-10
		```
		sh scripts/train-resnet50-imagenet-end-to-end.sh
		```
	* (Image segmentation) U-Net + Brain LGG
		```
		sh scripts/train-unet-brain-end-to-end.sh
		```
* note: for the baselines, we provide sample scripts [here](https://github.com/hylo-dso/kfac-pytorch) (based on the official release of KAISA [1](https://github.com/gpauloski/kfac-pytorch) and [2](https://github.com/HQ01/kfac_pytorch/tree/unet)).

## Analysis
* To enable profiling, add ```--profiling``` to the training command:
	* e.g. ```sh scripts/train-resnet50-imagenet-end-to-end.sh --profiling```
	* the outputs are written to .csv files

* To enable rank analysis, add ```--rank-analysis``` to the training command:
	* ```sh scripts/train-resnet32-cifar10-end-to-end.sh --rank-analysis```
	* the outputs are written to .csv files

* To check the gradient norm trend throughout the training, add ```--sngd``` and ```--grad-norm``` to the training command:
	* e.g. ```sh scripts/train-resnet32-cifar10-end-to-end.sh --sngd --grad-norm```
	* the outputs are written to ```grad-norm.csv```

* To check the gradient error of KID and KIS, add ```--grad-error``` and ```--enable-id```/```--enable-is``` to the training command:
	* e.g.
		```
		sh scripts/train-resnet50-imagenet-end-to-end.sh --grad-error --enable-id
		sh scripts/train-resnet50-imagenet-end-to-end.sh --grad-error --enable-is
		```
	* the outputs are written to ```grad-error-*.txt``` files


