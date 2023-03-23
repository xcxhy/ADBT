import os
import sys
sys.path.append("./ADBT")
from util import *
from pathlib import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

cateid_to_catename_dict = read_pkl('')
cates = [cateid_to_catename_dict[key][1] for key in cateid_to_catename_dict.keys()]

count_vect = CountVectorizer()

bow = count_vect.fit_transform(cates)

terms = count_vect.get_feature_names_out()

model = KMeans(n_clusters=10, init='k-means++', random_state=99)

model.fit(bow)

labels = model.labels_
cluster_center = model.cluster_centers_
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

for i in range(10):
    print("cluter %d:" %i, end="")
    for ind in order_centroids[i, :10]:
        print("%s" % terms[ind], end="")
        print()
print("END!")