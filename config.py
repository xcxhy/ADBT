import os
import sys
from sacred import Experiment

ex = Experiment("Blip2")

@ex.config
def config():
    model_name="blip2_t5"
    model_type="pretrain_flant5xl"
    embedd_name="blip2_feature_extractor"
    embedd_type="pretrain"
    mode="process"
    product_embedd_path="/data/xcxhy/LAVIS-main/examples/retrivael/data/product_embed.pt"
    product_text_dir="/data/xcxhy/LAVIS-main/examples/retrivael/data/captions.txt"
    product_img_dir="/data/xcxhy/LAVIS-main/examples/retrivael/data/image_url.txt"

