* `get_matbench_data.py` downloads tasks from matbench and saves them as `.csv` files in the directory `matbench_datasets`
* We used [CBFV](https://github.com/Kaaiian/CBFV) to featurize data. So we kept the .csv file that contains word embedding in the directory `cbfv/cbfv/element_properties`.
* `train_models.py` trains ML models for matbench tasks that are featurized using mat2vec, onehot encoding and our embedding vectors.
* `get_plots.py` generates figures of mae vs. percentage of dataset as shown in the publication.