# Deep Gaussian mixture model for unsupervised image segmentation

This repository is created for learning non-linear regularizing filters for inverting the Radon transform. For more detail about non-linear regularizing filters see:

```
Schwab, M., Mayr, A., & Haltmeier, M. (2024). Deep Gaussian mixture model for unsupervised image segmentation. arXiv preprint arXiv:2404.12252.
```


# Instalation

1. Clone the git repository. 
```
git clone https://git.uibk.ac.at/c7021123/deepGMM.git
``` 

2. Intall and activate the virtual environment.
```
cd deepGMM
conda env create -f env_deepG.yml
conda activate deepG
``` 

# Usage

## Preprocessing
1. Download the [MyoPS Dataset](https://mega.nz/folder/BRdnDISQ#FnCg9ykPlTWYe5hrRZxi-w) and save it in a new folder called `DATA`. After download the folder structure should look like this:
``` 
DATA/
├── MyoPS 2020 Dataset 
    ├── train25
    ├── train25_myops_gd
    ├── test20
    ├── MyoPS2020_EvaluateByYouself

```
Note that for the experiments of the paper only the folders train and train25_myops_gd are necessary. The other folders are optional and don't have to be present.

2. Prepare the downloaded dataset for the segmentation task. For this run the following command in your console
```
python preprocessing.py 
``` 
The preprocessed data will be saved as numpy files in the `DATA` folder of the repository.  

## Train deep Gaussian mixture models (deepG)

### deepG for single images

To train the two dimensional U-Nets run the command
```
python 2d-net.py DATASET_NAME FOLD 
``` 
- `FOLD` specifies on which on which od the five folds (0,1,2,3,4) the network should be trained.  
- Trained networks and training progress will get saved in a folder called `RESULTS_FOLDER` located in the `DATA_FOLDER` directory. 

### 2D-3D cascade

To train the Error correcting 2D-3D cascaded framework run the command
```
python 3d-cascade.py DATASET_NAME FOLD 
``` 
Note that to be able to train the cascade the 2D U-Net had to be trained beforehand. 

## Testing

To test the final framework on the testsets of the correspond of the EMICED callenge run 
```
python inference.py DATASET_NAME

```
- Note that inference will be done with all 5 folds from the cross-validation as an ensemble. Thus, 2D and 3D cascade must have been trained on all 5 folds prior to running inference.   
- Predictions and plots of the results will be saved in `RESULTS_FOLDER/DATASET_NAME/inference`.


## Authors and acknowledgment
Matthias Schwab<sup>1</sup>, Mathias Pamminger<sup>1</sup>, Christian Kremser <sup>1</sup>, Daniel Obmann <sup>2</sup>, Markus Haltmeier <sup>2</sup>, Agnes Mayr <sup>1</sup>

<sup>1</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria 

<sup>2</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria



