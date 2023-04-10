## Text translation

We use Facebook's open source project [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb). AI breakthrough project that open-sources models capable of delivering high-quality translations directly between any pair of 200+ languages — including low-resource languages like Asturian, Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere, regardless of their language preferences.

In fact, there are models of different sizes to choose from. Here we choose "facebook/nllb-200-1.3B".Translated into Chinese by default.If you don't change the model, you can directly execute the following code:
```
cd ./Text_only/text_translation
python translate_test.py
```
you can change the mode to "EN", and change the model name, like'facebook/nllb-200-distilled-600M'.

If the network is not good, you can go to the hugging face to download the corresponding model to the local.


