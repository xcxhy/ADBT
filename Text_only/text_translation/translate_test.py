import os
from translate import Translation

trans = Translation("facebook/nllb-200-distilled-600M")

while True:
    text = input("please enter your sentence: ")
    trans_text = trans.translate_zh(text)
    print(trans_text)

