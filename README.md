# Deep Gaussian mixture model for unsupervised image segmentation

This repository is created for unsupervised segmentation of multi-sequence MR images by combining Gaussian mixture models and deep learning techniques. For more details see:

```
Schwab, M., Mayr, A., & Haltmeier, M. (2024). Deep Gaussian mixture model for unsupervised image segmentation. arXiv preprint arXiv:2404.12252.
```


# Instalation

1. Clone the git repository. 
```
git clone https://github.com/matthi99/deepGMM.git
``` 

2. Intall and activate the virtual environment.
```
cd deepGMM
conda env create -f env_deepG.yml
conda activate deepG
``` 

# Usage

## Preprocessing
1. Download the [MyoPS Dataset](https://mega.nz/folder/BRdnDISQ#FnCg9ykPlTWYe5hrRZxi-w) and save it in a folder called `DATA`. After download the folder structure should look like this:
``` 
DATA
├── MyoPS 2020 Dataset
    ├── train25
    ├── train25_myops_gd
    ├── test20
    ├── MyoPS2020_EvaluateByYouself

```
Note that for the experiments of the paper only the folders `train25` and `train25_myops_gd` are necessary. The other folders are optional and don't have to be present.

2. Prepare the downloaded dataset for the segmentation task. For this run the following command in your console:
```
python preprocessing.py 
``` 
The preprocessed data will be saved as numpy files in `DATA/preprocessed/myops_2d/`.  

## Train deep Gaussian mixture models (deepG)

### deepG for single images

To apply the proposed methods deepG and deepSVG to the multi-sequence MR images of the MyoPS Dataset run
```
python deepG.py --type "TYPE" --lam XXX --tol XXX --max_epochs XXX
``` 
- `--type` specifies which Gaussian mixture model (GMM) should be used. You can decide between the classical GMM ("deepG") and the spacially variant GMM ("deepSVG"). Default setting is "deepG". 
- `--lam` specifies the regularization parameter (`default=0`) for the regularizing function $r(\bm{\mu})$ described in the paper. 
- `--tol` defines the stopping criteria. If the negative log-likelihood (NLL) change per iteration gets smaller than `tol` the algorithm is stopped. (`default=0.001`).  
- `--max_epochs` defines for how many epochs the networks should be trained maximally (`default=200`).

Predicted segmentation masks and results compared to the ground truth will be saved in the `RESULTS_FOLDER`.

### Training deepG on multiple images

To train a deep Gaussian mixture model on multiple images of the dataset run
```
python deepG_train.py --type "TYPE" --lam XXX --tol XXX --max_epochs XXX
``` 
The network will be trained on images of 20 patients following the same data split as in the paper. After training the performance on a test dataset of 5 patients can be obtained by running
```
python deepG_pred.py --type "TYPE" --lam XXX
``` 
Segmentation masks and results compared to the ground truth again will be saved in the `RESULTS_FOLDER`.


## Comarison with EM-algorithm

To compare the results of the proposed method with conventional NLL estimation with the EM-algorithm you can run
```
python EM_GMM.py --mu_data T/F --tol XXX --max_iter XXX

```
or 
```
python EM_SVGMM.py --mu_data T/F --tol XXX --max_iter XXX

```
- `--mu_data` specifies if $\bm{\mu}_{\text{data}}$ should be used as initalization for the parameter $\bm{\mu}$ (`default=False`).
- `--tol` defines the stopping criteria. If the NLL improvement per iteration gets smaller than `tol` the algorithm is stopped (`default=0.001`).  
- `--max_iter` defines maximal number of EM-iterations that are performed (`default=100`).

Predicted segmentation masks and results compared to the ground truth will be saved in the `RESULTS_FOLDER`.


## Authors and acknowledgment
Matthias Schwab<sup>1</sup>, Agnes Mayr <sup>1</sup>, Markus Haltmeier <sup>2</sup>

<sup>1</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria 

<sup>2</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

This work was supported by the Austrian Science Fund (FWF) [grant number DOC 110].

