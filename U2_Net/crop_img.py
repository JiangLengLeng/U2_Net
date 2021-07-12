import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import numpy as np
from PIL import Image
import glob
import cv2
from PIL import Image

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop(img,mask):
    # name, *_ = img_file.split(".")
    img_array = np.array(img)
    mask = np.array(mask)


    #通过将原图和mask图片归一化值相乘，把背景转成黑色
    #从mask中随便找一个通道，cat到RGB后面，最后转成RGBA
    # res = np.concatenate((img_array * (mask/255), mask[:, :, [0]]), -1)
    # print(res.shape)
    res = np.concatenate((img_array * (mask/255), mask[:,:,[0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    # img.show()
    return img

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


trans=transforms.Compose([transforms.Resize((120,160)),transforms.ToTensor()])


def output(image,pred):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    return imo


if __name__ == "__main__":

    model_name = 'u2net'  # u2netp
    model_dir = './saved_models/' + model_name + '/' + model_name + '.pth'
    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))

    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    cap1 = cv2.VideoCapture(r"http://ivi.bupt.edu.cn/hls/cctv2.m3u8")
    cap2 = cv2.VideoCapture(0)
    while True:
        ret1,frame1 = cap1.read()
        ret2,frame2 = cap2.read()
        frame1_Image = Image.fromarray(frame1)
        frame2_I = Image.fromarray(frame2)
        d1, d2, d3, d4, d5, d6, d7 = net(trans(frame2_I)[None, ...].cuda())
        frame2 = crop(frame2,output(frame2, normPRED(d1[:, 0, :, :])))
        frame2 = frame2.resize((150,200))
        frame1_Image.paste(frame2,(500,400),mask=frame2)
        frame = np.array(frame1_Image)
        cv2.imshow("img",frame)
        if cv2.waitKey(42) & 0xFF == ord("q"):
            break


