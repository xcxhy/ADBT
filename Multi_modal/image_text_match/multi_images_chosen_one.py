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
    
    def choose_one_image(self, )



ex.automain
def main(_config):
