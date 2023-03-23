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

