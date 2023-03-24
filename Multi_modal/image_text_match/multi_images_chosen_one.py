import os
import shutil
import torch
from pathlib import *
from PIL import Image
from config import ex
from lavis.models import load_model_and_preprocess


class Image_chonsen_One(object):
    def __init__(self,model_name, model_mode):
        self.model_name = model_name
        self.model_mode = model_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.text_processors = \
            load_model_and_preprocess(self.model_name, self.model_mode, device=self.device, is_eval=True)
    
    def choose_one_image(self, image_path, caption):
        raw_image = Image.open(image_path).convert("RGB")
        img = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        txt = self.text_processors["eval"](caption)
        itm_output = self.model({"image" : img, "text_input" : txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        itc_socre = self.model({"image": img, "text_input": txt}, match_haed="itc")
        return (image_path, itm_scores[:, 1].item(), itc_socre.item())

    



ex.automain
def main(_config):
