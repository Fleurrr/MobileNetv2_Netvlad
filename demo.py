import argparse
import torch
import torchvision
import os
import time
import numpy as np
import cv2
import glob
import h5py
import netvlad

def read_img(img_path, cuda):
    img = cv2.imread(img_path)
    img = img.astype('float32')
    C, H, W = img.shape[2], img.shape[1], img.shape[0]
    inp = img.copy()
    inp = (inp.reshape(C, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, C, H, W)
    if cuda:
        inp = inp.cuda()
    return inp

def load_model(weight_path ,cuda):
    encoder_dim = 1280
    encoder = torchvision.models.mobilenet_v2(pretrained=False)
    layers = list(encoder.features.children())
    encoder = torch.nn.Sequential(*layers)
    model = torch.nn.Module() 
    model.add_module('encoder', encoder)
    net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False)
    model.add_module('pool', net_vlad)
    
    if cuda:
        checkpoints = torch.load(weight_path)
        model.load_state_dict(checkpoints['state_dict'])
        model = model.cuda()
    else:
        checkpoints = torch.load(weight_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoints['state_dict'])
    model.eval()
    return model
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MobileNet_v2-Netvlad Demo.')
    parser.add_argument('--image', type=str, default='sample.png',
        help='Image file.')
    parser.add_argument('--cuda', action='store_true',
        help='Use cuda GPU to speed up network processing speed (default: False)')
    opt = parser.parse_args()
        
    cuda = opt.cuda
    img_path = opt.image
    if(cuda):
        print('=>Using GPU!')
    else:
        print('=>Using CPU!')
    
    inp = read_img(img_path, cuda)
    print('==>Successfully load the picture!')
    
    s1 = time.time()
    weight_path='pretrain-model.pth.tar'
    model = load_model(weight_path ,cuda)
    print('===>Successfully load the network! Using time:', time.time()-s1)
    
    s2 = time.time()
    image_encoding = model.encoder(inp)
    vlad_encoding = model.pool(image_encoding) 
    print('====>Infer time:', time.time()-s2)  
    
    
