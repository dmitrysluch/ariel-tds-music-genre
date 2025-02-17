# Audio genre detection

The task of audio sample genre detection is usually solved using neural networks. I am trying to develop an alternative approach using classical ML algorithm (gradient boosting over decision trees to be precise) and manual feature extraction and selection. The dataset used is GTZAN consisting of 1000 30-second tracks of ten genres. The best known accuracy on this dataset (using NNs) is `0.95`, I achieve score of `0.81`.

The notebook with the solution is called `main.ipynb`


The dagxtractor.py is map-reduce library written by me because I got out of RAM

model.py is a wrapper around catboost model