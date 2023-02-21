import os
import sys
import torch
import requests
import argparse
# import keyboard
from PIL import Image
from sacred import Experiment
from lavis.models import load_model_and_preprocess

ex = Experiment('Infer')

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class VQA_INFER(object):
    def __init__(self, args):
        self.name = args['name']
        self.model_type = args['model_type']
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=self.name, \
                                                                      model_type=self.model_type, \
                                                                        is_eval=True, device=device) 
                                                                                
        self.save_path = args['save_path']
        # self.template = self.init_template()
        # if args["text_only"] == 0:
        # self.image = self.get_image(args["img_url"])
        
    def get_image(self, img_url):
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB') 
        # raw_image= Image.open(img_url).convert("RGB")
        return self.vis_processors['eval'](raw_image).unsqueeze(0).to(device)
      
    def init_template(self, content):
        template = "Question: {} Answer: {}."
        init_quesition = "what is the specific information of this product?"
        return template.format(init_quesition, content)

    def get_input(self):
        question = input("请输入你的问题:")
        return question
    
    def get_generate(self, template, image, question):
        prompt = template + " Question: " + question + " Answer:"
        answer = self.model.generate({"image": image, "prompt" : prompt}, use_nucleus_sampling=False,)
        template = prompt + answer[0]
        return answer[0], template
    
    def infer(self):
        while True:
          image_url = input("请输入图片在线链接: ")
          if image_url=='exit':
              break
          content = input("请输入产品文本信息: ")
          with open(self.save_path, "a+") as f:
            f.write(content + "\n" )
            image = self.get_image(image_url)
            template = self.init_template(content)
            while True:
                # if keyboard.sent('esc'):
                #     break
                question = self.get_input()
                if question=='exit':
                    break
                answer, template = self.get_generate(template, image, question)
                print("Answer: " + answer)
                f.write("Question: "+ question + '\n')
                f.write("Answer: " + answer + '\n')
          
        return
    # def get_only_generate(self, question):
    #     prompt = self.template + " Question: " + question + " Answer:"
    #     answer = self.model.generate({"image": 0, "prompt" : prompt}, use_nucleus_sampling=False,)
    #     self.template = prompt + answer[0]
    #     return answer[0]

# @ex.config
# def config():
#     name="blip2_t5"
#     model_type="pretrain_flant5xl"
#     # text_only=0
#     img_url = "https://image.made-in-china.com/2f1j00ODaofYwPZHqJ/Personalized-Sportswear-Custom-Team-Logo-Breathable-Polyester-Baseball-Jersey.jpg"
#     pd_content="Personalized Sportswear Custom Team Logo Breathable Polyester Baseball Jersey \
#       Reference FOB Price Get Latest Price  US $6.5-16.5 / Piece | 1 Piece (Min. Order) \
#         Age:	Youth & Adults \
#           Gender:	Unisex \
#             Material:	Polyester / Spandex \
#               Type:	Top \
#                 Feature:	Breathable, Quick-Drying, Moisture-Wicking, Anti-UV \
#                   Usage:	Ball Sports, Fitness, Track and Field, Baseball/Softball \
#                     Size: Xs-6XL Custom Service: Name, Logo, Number, Pattern \
#                       Sizes: Customized Logo: Custom Logo Production Capacity: 50000pieces/Month \
#                         Product Type: Sportswear Printing: Custom Sublimation Printing \
#                           Age Group: Adult, Youth Color: Any Color Leading Time: 7-15 Days \
#                             Specification: 1pcs/set in one poly bag, 30-100pcs in one carton \
#                               Origin: China"
    
    
#     save_path="/data/xcxhy/LAVIS-main/examples/result.txt"
# _config = config()
# GE = VQA_INFER(_config)
# @ex.automain
# def main(_config):
#     while True:
#       image_url = input("请输入图片在线链接: ")
#       content = input("请输入产品文本信息: ")
#       with open(_config['save_path'], "a+") as f:
#           f.write(content + "\n" )
#           image = GE.get_image(image_url)
#           template = GE.init_template(content)
#           while True:
#                 # if keyboard.sent('esc'):
#                 #     break
#                 question = GE.get_input()
#                 if question=='esc':
#                     break
#                 answer, template = GE.get_generate(template, image, question)
#                 print("Answer: " + answer)
#                 f.write("Question: "+ question + '\n')
#                 f.write("Answer: " + answer + '\n')
    # with open(_config['save_path'], "a+") as f:
    #     f.write(_config["pd_content"] + '\n')
    #     while True:
    #         question = GE.get_input()
            
    #         answer = GE.get_generate(question)
    #         print("Answer: " + answer)
    #         f.write("Question: "+ question + '\n')
    #         f.write("Answer: " + answer + '\n')
