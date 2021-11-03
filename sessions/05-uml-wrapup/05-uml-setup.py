######################################################
## Wrap up UML
## Hands on-heavy class to dive into some concepts you may want to explore 
## Learning objectives:
##
## 0. wrap up PCA
## 1. exposure to more contemporary techniques for interviews, awareness, and further exploration
## 2. highlight different use-cases, some that help with viz for non-tech, others to think about alternatives to linear PCA
## 3. use this as a jumping off point for you to consider the fact that are lots of methods, and its not typically for this task, do this one approach
######################################################

## resources
# - https://pair-code.github.io/understanding-umap/
# - https://distill.pub/2016/misread-tsne/
# - repo has a resources folder, review the about.md file for additional links.


# installs
# ! pip install umap-learn
# pip install umap-learn

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# scipy
from scipy.spatial.distance import pdist

# scikit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

# GET ON WITH THE PCA
# 1. get the wine dataset from Big Query
# 2. our ultimate goal is to predict the quality column
# 3. standardize the columns with min max scaler (tricky) <- for practice purposes
# 4. fit a PCA model to the dataset
# 5. plot the variance explained (your choice)
# 6. OPTIONAL fit a simple decision tree to predict the quality column

# 1
# this is my billing project
# NOTE:  You shoudl replace below with your own billing project
PROJECT_ID = 'ba-820-lt'


# make the query and get the data
SQL = "select * from `questrom.datasets.wine`"


# we can also do this via pandas 
df = pd.read_gbq(SQL, PROJECT_ID)
df.head() 
df.describe().T
df.shape

#Teacher's answer
X = df.drop(columns = 'quality')
y = df.quality

mm = MinMaxScaler()
Xs = mm.fit_transform(X)

pca = PCA()
pcs = pca.fit_transform(Xs)

plt.plot(range(pca.n_components_), pca.explained_variance_ratio_)
plt.show() #Right here we can see that we can keep up to 4 principle components

pc4 = pcs[:, :4]
pc4

np.cumsum(pca.explained_variance_ratio_)

tree = DecisionTreeClassifier(max_depth = 4, min_samples_split = 20, min_samples_leaf = 7)
tree.fit(pc4, y)
preds = tree.predict(pc4)
preds[:4]
metrics.accuracy_score(y, preds)

#3
scaler = MinMaxScaler()
print(scaler.fit(df))
print(scaler.data_max_)
print(scaler.transform(df))

#4
# Correlation matrix
wcor = df.corr()
sns.heatmap(wcor, cmap ='Reds', center = 0)
plt.show()

# Fit our PCA model
pca = PCA() # so we set that it will explain 90% of the variance of the data
pcs = pca.fit_transform(df)
type(pcs)
pcs.shape

pcs[:5, :5]

# variance explanation ratio -- pc expalined Variance
varexp = pca.explained_variance_ratio_
varexp
type(varexp)
varexp.shape
np.sum(varexp)

#Plot the variance explained the PC
plt.title('Explained variance per PC')
# sns.lineplot(range(1, len(varexp) + 1), varexp)
sns.lineplot(range(1, len(varexp) + 1), np.cumsum(varexp))
plt.axhline(.95) #This is the reference line
plt.show()
##############################################################################
## Code samples
##############################################################################


############################################ mnist data
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
X.shape #1797,64, and each one of the feature is a pixel
img = digits.images[0]
img.shape #8,8,8

#Let's plot this image
plt.imshow(img, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.show()
#black and white only have 1 channel, while red green blue has 3 channels

img.flatten()

#plot a heat map 
cor = X.corr()
sns.heatmap(cor, cmap ='Reds', center = 0)
plt.show()


############################################ decision tree
from sklearn.tree import DecisionTreeClassifier  # simple decision tree
tree = DecisionTreeClassifier(max_depth=5)   # max depth of tree of 4 is random for example
tree.fit(X, y)  # sklearn syntax is everywhere!
tree_preds = tree.predict(X)   # 
tree_acc = tree.score(X, y)
tree_acc


# two step process - use PCA to reduce, then tsne for embeddings(2d)
# to keep things simple, just random choice for 90%, this could be 80% or 95% of
pca_m = PCA(.9)
pca_m.fit(X)
pcs_m = pca_m.transform(X)

np.sum(pca_m.explained_variance_ratio_)

############################################ tsne
tsne = TSNE()
tsne.fit(pcs_m)
te = tsne.embedding_
te.shape

tdata = pd.DataFrame(te, columns=["e1", "e2"])
tdata['y'] = y

tdata.head()


############################################ seaborn mnist plot
PAL = sns.color_palette("bright", 10) 
plt.figure(figsize=(10, 8))
sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)
plt.show()
#we can see that there are still errors, maybe this error is from the labelling data

X2 = tdata.drop(columns = 'y')
y2 = tdata.y

tree2 = DecisionTreeClassifier(max_depth = 5)
tree2.fit(X2, y2)
tree2_preds = tree2.predict(X2)
tree2_acc2 = tree2.score(X2, y2)
tree2_acc2



############################################
import umap.umap_ as UMAP

X = digits.data
y = digits.target

u = UMAP(random_state=820, n_neighbors=10)
u.fit(X)
embeds = u.transform(X)

# put it onto a dataframe
umap_df = pd.DataFrame(embeds, columns = ['x', 'y'])
umap_df['y'] = y

umap_df.head(3)

# plot
sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)
plt.show()




##############################################################################
## Other Considerations for UML
##############################################################################
##
##  Other iterations of PCA even
##     - Randomized PCA (generalizes and approximates for larger datasets)
##     - Incremental PCA (helps when the data can't fit in memory)
##
##  Recommendation Engines
##      - extend "offline" association rules 
##      - added some links to the resources (great article with other libraries)
##      - toolkits exist to configure good approaches for real-use
##      - I call reco engines unsupervised because its moreso about using neighbors and similarity to back 
##        into items to recommend
##      - can be done by finding similar users, or similar items.
##      - hybrid approaches work too
##      - scikit surprise
##      NOTE:  Think about it? you can pull data from databases!  Build your own reco tool by running a simple API!
##             batch calculate recos and store in a table, send user id to API, look up the previously made recommendations
##             post feedback to database, evaluate, iterate, repeat!
##     
##   A python package to review
##      - I followed this package in its early days (graphlab) before Apple bought the company
##      - expressive, with pandas-like syntax
##      - accessible toolkit for a number of ML tasks, including Reco engines 
##      - https://github.com/apple/turicreate



################### Breakout Challenge
## work as a group to combine UML and SML!
## housing-prices tables on Big Query will be used questrom.datasets._______
##     housing-train = your training set
##     housing-test = the test set, does not have the target, BUT does have an ID that you will need for your submission
##     housing-sample-submission = a sample of what a submission file should look like, note the id and your predicted value
## 
## use regression to predict median_house_value
## 
## you can use all of the techniques covered in the program, and this course
## objective:  MAE - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
##
##
## tips/tricks
##    - ITERATE!  iteration is natural, and your friend
##    - submit multiple times with the same team name
##    - what would you guess without a model, start there!
##    - you will need to submit for all IDs in the test file
##    - it will error on submission if you don't submit 
##
## Leaderboard and submissions here: TBD
## 