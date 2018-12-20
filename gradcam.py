import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import pdb

class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            classifier=['linear']
            features=['act1', 'act2', 'bn1', 'bn2','layers','conv1','conv2' ]

            for name, module in self.model.module.named_children():
                #pdb.set_trace()
                #if name == 'classifier':
                if name in classifier:
                    #pdb.set_trace()
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
                    feature = F.avg_pool2d(feature, 6)
                    feature = feature.view(feature.size(0), -1)
                    feature = module(feature)
                elif name in features:
                    feature = module(feature)
                #print(name)
                #if name == 'features':
                #if name in features:
                    #pdb.set_trace()
                    #feature.register_hook(self.save_gradient)
                    #self.feature = feature
                    #if name =='act2':
                    #    feature = F.avg_pool2d(feature, 6)
                        #feature = feature.view(feature.size(0), -1)
            #pdb.set_trace()
            classes = F.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()
            #pdb.set_trace()
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)