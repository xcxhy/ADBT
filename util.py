import os
import json
import pickle

def read_pkl(path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except:
        print('Read pkl Error!')
        return 

def write_pkl(path, data):
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print("Write {} end!".format(path))
        return 
    except:
        print("Write pkl Error!")
        return