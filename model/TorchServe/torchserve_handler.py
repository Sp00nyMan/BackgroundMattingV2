import os.path

import torch

from model import MattingRefine
from torchvision.transforms import Compose, ToTensor
from ts.torch_handler.base_handler import BaseHandler

class Model(BaseHandler):
    image_processing = Compose([ToTensor])

    def __init__(self):
        self.initialized = False
        self._context = None
        self.device = None
        self.model = None

    def initialize(self, args):
        self.model = MattingRefine('resnet101',
                                   0.25,
                                   'sampling',
                                   80_000,
                                   0.7,
                                   3)
        properties = args.system_properties
        self.device = torch.device(f"cuda:{str(properties.get('gpu_id'))}" if torch.cuda.is_available() else "cpu")
        model_pth = os.path.join(properties.get('model_dir'), args.manifest['model']['serializedFile'])
        if not os.path.isfile(model_pth):
            raise RuntimeError("pth file is missing")

        self.model = self.model.to(self.device).eval()
        self.model.load_state_dict(torch.load(model_pth, map_location=self.device), strict=False)
        self.initialized = True

    def inference(self, input):
        src, bgr = input
        src = src.to(self.device, non_blocking=True)
        bgr = bgr.to(self.device, non_blocking=True)
        pha, fgr, *_ = self.model(src, bgr)
        return pha, fgr

    def handle(self, data, context):
        return self.inference(data)
