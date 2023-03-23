import os
import shutil
import torch
from pathlib import *
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, text_processors = load_model_and_preprocess("")