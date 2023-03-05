import os
import sys
import math
import torch
import faiss
import requests

from util import *
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device('cuda:4' if torch.cuda.is_available() else "cpu")

class Retrieval(object):
    def __init__(self, args):
        self.model_name = args["model_name"]
        self.model_type = args["model_type"]
        self.embedd_name = args["embedd_name"]
        self.embedd_type = args["embedd_type"]
        self.mode = args["mode"]
        self.product_embeddings_path = args["product_embedd_path"]
        self.product_texts = read_list_txt(args["product_text_dir"])
        self.product_images = read_list_txt(args["product_img_dir"])

        self.embedd_model, self.emb_vis_processors, self.emb_txt_processors = \
            load_model_and_preprocess(name=self.embedd_name,  model_type=self.embedd_type, is_eval=True, device=device)
        self.generate_model, self.gen_vis_processors, self.gen_txt_processors = \
            load_model_and_preprocess(name=self.model_name,  model_type=self.model_type, is_eval=True, device=device)

        if self.mode == 'process':
            self.product_embeddings = self.embedd_product(self.product_images, self.product_texts)
        elif self.mode == "load":
            self.product_embeddings = torch.load(self.product_embeddings_path)
        self.index = self.load_retrieval(self.product_embeddings)
    
    # def get_image_url(self): # 访问私有实例属性 
    #     return self.product_images
    
    # def get_content(self):
    #     return self.product_texts
    
    # product_images = property(get_image_url)
    # product_texts = property(get_content)
    
    def embedd_product(self, images, texts):
        if len(images) != len(texts):
            print("Image and Texts is not pair!")
            return
        all_embeddings = []
        for i in range(len(images)):
            if i == 0:
                image, text = images[i], texts[i]
                raw_image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
                pro_image = self.emb_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                pro_text = self.emb_txt_processors["eval"](text)
                sample = {"image": pro_image, 'text_input' : pro_text}
                input_features = self.embedd_model.extract_features(sample).multimodal_embeds[:,0,:]
                all_embeddings = input_features
            else:
                image, text = images[i], texts[i]
                raw_image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
                pro_image = self.emb_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                pro_text = self.emb_txt_processors["eval"](text)
                sample = {"image": pro_image, 'text_input' : pro_text}
                input_features = self.embedd_model.extract_features(sample).multimodal_embeds[:,0,:]
            # all_embeddings
                all_embeddings = torch.cat([all_embeddings, input_features], dim=0)
        all_embeddings = np.array(all_embeddings.cpu().numpy())
        torch.save(all_embeddings, self.product_embeddings_path)
        return all_embeddings
    # def init_question(self, question):
    #     pro_question = self.emb_txt_processors["eval"](question)
    #     self.all_question = pro_question
    #     sample = {"text_input" : self.all_question}
    #     features_text = self.embedd_model.extract_features(sample, mode="text").text_embeds[:, 0]
    #     return np.reshape(np.array(features_text.cpu()), (1,768)).astype('float32')

    def embedd_question(self, index, question):
        pro_question = self.emb_txt_processors["eval"](question)
        if index == 0:
            self.all_question = pro_question
        else:
            self.all_question += pro_question
        sample = {"text_input" : self.all_question}
        features_text = self.embedd_model.extract_features(sample, mode="text").text_embeds[:, 0]
        return np.reshape(np.array(features_text.cpu()), (1,768)).astype('float32')
    
    def init_template(self, content):
        template = "Question: {} Answer: {}."
        init_quesition = "what is the specific information of this product?"
        template = template.format(init_quesition, content)
        self.template = template
        return 
    
    def get_image(self, img_url):
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB') 
        # raw_image= Image.open(img_url).convert("RGB")
        return self.gen_vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    
    def generate_answer(self, image, question):
        prompt = self.template + " Question: " + question + " Answer:"
        #  answer = self.model.generate({"image": image, "prompt" : prompt}, use_nucleus_sampling=True,)
        answer = self.generate_model.generate({"image": image, "prompt" : prompt}, use_nucleus_sampling=False,)
        # self.template = prompt + answer[0]
        return answer[0]

    def load_retrieval(self, labels):
        # labels = np.array(labels).astype("float32")
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(768)
        index = faiss.index_cpu_to_gpu(res, 1, index)
        index.add(labels)
        return index
    
    def retrieval(self, input, k=3):
        D, I = self.index.search(input, k)
        I = np.squeeze(I,0)
        retrieval_product = [(self.product_images[i], self.product_texts[i]) for i in I]
        return retrieval_product

