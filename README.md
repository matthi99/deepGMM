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

To apply the proposed methods deepG and deepSVG to the multi-sequence MRI images of the MyoPS Dataset run
```
python deepG.py --type "TYPE" --lam XXX --tol XXX --max_epochs XXX
``` 
- `--type` specifies which Gaussian mixture model (GMM) should be used. You can decide between the classical GMM (deepG) and the spacially variant GMM (deepSVG). Default setting is `deepG`. 
- `--lam` specifies the regularization parameter (`default=0`) for the regularizing function $r(\mathbb{\mu})$ described in the paper. 
- `--tol` defines the stopping criteria. If the negative log-likelihood (NLL) change per iteration gets smaller than `tol` the algorithm is stopped. (`default=0.001`).  
- `--max_epochs` defines for how many epochs the networks should be trained maximally (`default=200`)

Predicted segmentation masks and results compared to the ground truth will be saved in the `RESULTS_FOLDER`

### Training deepG on multiple images

To train a deep Gaussian mixture model on multiple images of the dataset run
```
python deepG_train.py --type "TYPE" --lam XXX --tol XXX --max_epochs XXX
``` 
The network will be trained on images of 20 patients following the same data split as in the paper. After training the performance on a test dataset of 5 patients can be obtained by running
```
python deepG_pred.py --type "TYPE" --lam XXX --tol XXX --max_epochs XXX
``` 
Segmentation masks and results compared to the ground truth again will be saved in the `RESULTS_FOLDER`


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



