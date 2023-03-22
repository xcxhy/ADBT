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
    
def read_list_txt(path):
    with open(path, "r") as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip("\n")
    print("Read {} List End!".format(path))
    return data

def write_list_txt(path, data):
    with open(path, 'w') as f:
        for value in data:
            f.write(value + '\n')
    print("Write {} List End!".format(path))
    return 

def read_dict_json(path):
    with open(path, 'r+') as file:
        content=file.read()
    content=json.loads(content)
    return content

def write_dict_json(path, data):
    dict_json = json.dumps(data, indent=4)
    with open(path, 'w') as file:
        file.write(dict_json)
    return