# Word embedding vectors for materials, learned from Robocrystallographer descriptions

This code generates structure based feature vectors derived from Robocrystallographer text descriptions
of crystal structures using word embeddings. It uses [`mat2vec`](https://github.com/materialsintelligence/mat2vec) to learn these embeddings from Robocrystallographer descriptions for all the elements in periodic table.

## Table of Contents

* Installation
* Reproduction of publication results
* Publications/How to cite
* Mainteiners

## Installation and basic usage

### Clone or download this GitHub repository

Do any one of the following:

* Clone this [repository](https://github.com/hasan-sayeed/robo_descriptor) to a directory of your choice on your computer.
* Download an archive of this [repository](https://github.com/hasan-sayeed/robo_descriptor) and extract it to a directory of your choice on your computer.

### Install dependencies

* Make sure you have the pip module installed.
* Navigate to the root folder of this repository and run `pip install --ignore-installed -r requirements.txt`. Note: If you are using a conda environment and any packages fail to compile during this step, you may need to first install those packages separately with `conda install package_name`.
* You are ready to go!

## Reproduction of publication results

* To get robocrystallographer description for stable materials from materials project, navigate to the directory `get_embeddings` and run `mpid_to_robo_descriptions.py`.
* You can train [mat2vec](https://github.com/materialsintelligence/mat2vec) on your downloaded corpus of robocrystallographer description and generate word embeddings for all the elements of periodic table. `get_embeddings.py` in `get_embeddings` directory contains directions for that.
* To download matbench dataset navigate to `training` directory and run `get_matbench_data.py`.
* You can train ML models on matbench tasks while featurizing the data using our word embedding vectors by running `train_models.py` from the directory `training`. We used [composition based feature vector](https://github.com/Kaaiian/CBFV) technique for the featurization. You need to keep the .csv file that contains word embedding in the directory `training/cbfv/cbfv/element_properties`.
* To plot figures as shown in the paper run `get_plots.py` in the same directory. This will save the figures in the directory `results`.

## Publications/How to cite

## Mainteiners

[hasan-sayeed](https://github.com/hasan-sayeed) (main maintainer)