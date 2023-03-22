import os
import sys
sys.path.append("./ADBT")
from util import *
from pathlib import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

cateid_to_catename_dict = read_pkl()