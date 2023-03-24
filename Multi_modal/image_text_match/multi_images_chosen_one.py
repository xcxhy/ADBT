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

    def get_best_score(self, result, itm_paramerter, itc_parameter):
        itm_sort = sorted(result, key=lambda x: x[1], reverse=True)
        itc_sort = sorted(result, key=lambda x: x[2], reverse=True)

        itm, itc = itm_sort[0], itc_sort[0]

        if itm[1] > itm_paramerter:
            return itm[0]
        else:
            return 0
    def save_best_image(self, save_path, image_path):
        if image_path == 0:
            print("No match images!")
            return
        image_name = image_path.split("/")[-1]
        if Path(save_path).exists():
            shutil.copy(image_path, os.path.join(save_path, image_name))
        else:
            os.mkdir(save_path)
            shutil.copy(image_path, os.path.join(save_path, image_name))
        
        return




ex.automain
def main(_config):
    IcO = Image_chonsen_One(_config["embedd_name"], _config["embedd_type"])
    dir_path = ""
    save_path = os.path.join(dir_path, "result")
    product_path = list(os.walk(dir_path))
    images_path = product_path[0][2]

    caption= "Head Light"
    text_image_pair = []
    for img_dir in images_path:
        text_image_pair.append((os.path.join(dir_path, img_dir), caption))
    
    res = list(map(IcO.choose_one_image, *zip(*text_image_pair)))
    res_path = IcO.get_best_score(res, 0.5, 0.5)
    IcO.save_best_image(save_path, res_path)

