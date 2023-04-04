import os
import argparse
from translate import Translation

def trans(name,mode):
    trans = Translation(name)
    if mode == "ZH":
        while True:
            text = input("please enter your sentence: ")
            trans_text = trans.translate_zh(text)
            print(trans_text)

    elif mode == "EN":
        while True:
            text = input("please enter your sentence: ")
            trans_text = trans.translate_en(text)
            print(trans_text)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_name', type=str, default="facebook/nllb-200-1.3B")
    parser.add_argument("--mode", type=str, default="ZH")
    args = parser.parse_args()
    trans(args.trans_name, args.mode)