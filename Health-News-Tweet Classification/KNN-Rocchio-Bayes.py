
import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import cross_val_score

file_name = 'dataset/health_tweets_labeled.csv'
df = pd.read_csv(file_name,encoding='ANSI')
df_sample = pd.DataFrame(df)
df.head()
data = np.array(df.iloc[:,0])
target = np.array(df.iloc[:,1])



file_name = 'dataset/health_tweets_labeled.csv'
df = pd.read_csv(file_name,encoding='ANSI')
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['tweet'].values.astype('U'))
arr=x.toarray()
vectorizer.vocabulary_.get('document')

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(arr)


# len(arr)
len(target)


# KNeighbors"

# print(tfidf)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(arr[:1500], target[:1500])
neigh.score(arr[1500:1600], target[1500:1600])

######################### Roccio ################################################



clf = NearestCentroid()
clf.fit(arr[:40000], target[:40000])
NearestCentroid(metric='euclidean', shrink_threshold=None)
print(clf.score(arr[50000:59000],target[50000:59000]))
scores = cross_val_score(clf, arr[50000:59000], target[50000:59000], cv=5)

print(scores)

len(target)

#################### Naive-bayes##########################################


# gnb = GaussianNB()
# y_pred = gnb.fit(arr[:500], target[:500]).predict(arr[500:600])
# print("Number of mislabeled points out of a total %d points : %d"
#       % (arr.shape[0],(target != y_pred).sum()))

clf = GaussianNB()
clf.fit(arr[:500], target[:500])
clf.score(arr[1000:1300],target[1000:1300])
clf_pf = GaussianNB()
clf_pf.partial_fit(arr[:500], target[:500], np.unique(target))
clf_pf.score(arr[3000:3300],target[3000:3300])
