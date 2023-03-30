from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class Translation(object):
    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
    
    def tokenize(self, text):
        token_text = self.tokenizer(text, return_tensors="pt")
        return token_text
    
    def translate_en(self, text):
        token = self.tokenize(text)
        translated_token = self.model.generate(**token, forced_bos_token_id=self.tokenizer.lang_code_to_id['eng_Latn'], max_length=1024)
        en_text = self.tokenizer.batch_decode(translated_token, skip_special_tokens=True)[0]
        return en_text
    def translate_zh(self, text):
        token = self.tokenize(text)
        translated_token = self.model.generate(**token, forced_bos_token_id=self.tokenizer.lang_code_to_id['zho_Hans'], max_length=1024)
        zh_text = self.tokenizer.batch_decode(translated_token, skip_special_tokens=True)[0]
        return zh_text

