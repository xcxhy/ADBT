import os
import argparse
from translate import Translation

def trans(name):
    trans = Translation(name)

    while True:
        text = input("please enter your sentence: ")
        trans_text = trans.translate_zh(text)
        print(trans_text)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans_name', type=str, default="facebook/nllb-200-1.3B")
    args = parser.parse_args()
    trans(args.trans_name)