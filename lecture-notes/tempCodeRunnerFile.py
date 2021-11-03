import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

# our dataset
# SQL = "SELECT * from `questrom.datasets.mtcars`"
# YOUR_BILLING_PROJECT = "questrom"
# cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)
cars = pd.read_csv('C:/Users/LinhTo/Desktop/BU/Class - Teacher/BA 820/BA820-Fall-2021/datasets/cars.csv')
