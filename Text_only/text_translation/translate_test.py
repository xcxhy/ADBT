import os
from translate import Translation

trans = Translation("facebook/nllb-200-1.3B")

while True:
    text = input("please enter your sentence: ")
    trans_text = trans.translate_zh(text)
    print(trans_text)

