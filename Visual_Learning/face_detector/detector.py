import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from res18_baseline import Res18Baseline
from res18_skip import Res18Skip
from torchvision import transforms
import cv2

class Detector:
    def __init__(self, ckpt, use_gpu=False, model='res18_baseline'):
        self.args = args
        if model == 'res18_baseline':
            self.model = Res18Baseline(args)
        elif model == 'res18_skip':
            self.model = Res18Skip(args)
        # self.model = Res18Baseline(args)
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        self.load_weights(ckpt)
        self.model = self.model.eval()
        cmd_printer.divider(text="warning")
        print('This detector uses "RGB" input convention by default')
        print('If you are using Opencv, the image is likely to be in "BRG"!!!')
        cmd_printer.divider()
        # color in bgr order
        self.colour_code = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                                    (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                    (0, 128, 128), (128, 128, 128), (64, 0, 0),
                                    (192, 0, 0), (64, 128, 0), (192, 128, 0)])

    def detect_single_image(self, np_img):
        torch_img = self.np_img2torch(np_img)
        tick = time.time()
        with torch.no_grad():
            pred = self.model.forward(torch_img)
            if self.use_gpu:
                pred = torch.argmax(pred.squeeze(),
                                    dim=0).detach().cpu().numpy()
            else:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        colour_map = self.visualise_output(pred)
        return pred, colour_map

    def visualise_output(self, nn_output):
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
        colour_map = np.stack([b, g, r], axis=2)
        colour_map = cv2.resize(colour_map, (256, 256), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 200)
        pad = 5
        labels = ['hair', 'face', 'bg']
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for i in range(0, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (int(c[2]), int(c[1]), int(c[0])), thickness=-1)
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (200, 200, 200), thickness=1)
            colour_map  = cv2.putText(colour_map, labels[i],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (200, 200, 200))
            pt = (pt[0], pt[1]+h+pad)
        return colour_map[..., ::-1]

    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path,
                              map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(256, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                        #                         saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        return img
