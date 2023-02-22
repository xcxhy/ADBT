import os
import sys
import torch
import requests
import argparse
# import keyboard
from PIL import Image
import streamlit as st
from sacred import Experiment
# from flask import Flask, request, render_tempalte
from infer import VQA_INFER
from config import ex
from lavis.models import load_model_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
_config = {'name':"blip2_t5", 'model_type':"pretrain_flant5xl", 'save_path':"/data/xcxhy/LAVIS-main/examples/result.txt"}

@st.cache_resource
def load_model():
    model, vis_processors, txt_processors = load_model_and_preprocess(name=_config['name'], \
                                                                      model_type=_config['model_type'], \
                                                                        is_eval=True, device=device)
    return model, vis_processors, txt_processors
def infer(model, vis_processors, txt_processors, raw_image, caption, question):
    # raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB') 
    image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
    template = "Question: {} Answer: {}."
    init_quesition = "what is the specific information of this product?"
    template = template.format(init_quesition, caption)
    prompt = template + " Question: " + question + " Answer:"
    answer = model.generate({"image": image, "prompt" : prompt}, use_nucleus_sampling=False,)
    return answer[0]

# def infer(model, image_url, caption,question):
#     image = model.get_generate(image_url)
#     if template==None:
#         template = model.init_template(caption)(allow_output_mutation=True, suppress_st_warning=True)
#     answer, template = model.get_generate(template, image, question)
#     return answer, template

st.title("This is a Product Robot!")
image_type = st.radio("What your image type?", ('image_url', "upload_image"))
if image_type== "image_url":
    image_url = st.text_input(label="Image URL")
    if image_url != "":
        origin_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB') 
        if st.button("Enter Image"):  
            st.image(origin_image, caption="porduct image")
else:
    image_file = st.file_uploader("Choose a image")
    if st.button("Upload Image"):
        origin_image = Image.open(image_file).convert('RGB') 
        st.image(origin_image, caption="porduct image")
# st.write('图片链接: ', image_url)
# with st.spinner("Wait for input!"):
# st.image(origin_image, caption="porduct image")
#         st.success("Done!")
# origin_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB') 
# if st.button("Enter image"):  
#     st.image(origin_image, caption="porduct image")
    # if st.button("enter product content"):
caption = st.text_input(label="Product Caption")
if st.button("Enter Caption"):    
    st.write('Caption', caption)

question = st.text_input("Question: ",value="")
if st.button("Enter Question"):
    model, vis_processors, txt_processors = load_model()
    answer = infer(model, vis_processors, txt_processors, origin_image, caption, question)
    st.write("ANSWER: ", answer)