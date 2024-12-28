import albumentations as A
import numpy as np
import cv2
from typing import Union
from pathlib import Path
from models.dif_unet import LitDimma
import torch
import matplotlib.pyplot as plt

def retinex_decomposition(img):
    R = img / (img.sum(axis=2, keepdims=True) + 1e-6)
    L = (img / (3 * R + 1e-6)).max(axis=2)
    return R, L

def pre_transform(img):
    light=img
    R, L = retinex_decomposition(light / 255.0)
    source_lightness = torch.tensor(L.mean()).float()
    hist_eq = A.augmentations.functional.equalize(light) / 255.0
    light = np.concatenate([light / 255.0, hist_eq, R, L[:, :, None]], axis=2).transpose(2, 0, 1)
    light = torch.from_numpy(light).float()
    return light,source_lightness
def read_image_cv2(path: Union[str, Path]) -> np.ndarray:
    """Read an image from a path."""
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Modle_Path = 'checkpoints/my_model/my_model-v2.ckpt'
    model = LitDimma.load_from_checkpoint(Modle_Path)

    # file_pathname = "pre_img/data"
    # img=cv2.imread()
    img = cv2.imread('pre_img/data/div_000028.png')
    img_high=cv2.imread('pre_img/output_img/div_000028.png')
    mse_ori=((img-img_high)**2).mean()#110
    print(mse_ori)
    # cv2.imshow("input", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_high=torch.from_numpy(img_high)
    input_img,l=pre_transform(img) #(10,400,400)
    l_target=l

    input_img_gpu=input_img.to(device)
    input_img_gpu=input_img_gpu.unsqueeze(0)
    l_gpu=l.to(device)
    l_target_gpu=l_target.to(device)
    # img_high_gpu=img_high.to(device)
    # print(input_img_gpu.shape)
    output_img=model(input_img_gpu,l_gpu,l_target_gpu)
    # print(output_img.shape)
    detached_output_img = output_img.detach()
    image_np = detached_output_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np_255=(image_np* 255).astype(np.uint8)
    output_img_cv2=cv2.cvtColor(image_np_255, cv2.COLOR_BGR2RGB)
    # print(output_img_cv2.shape)
    # print(img_high_gpu.shape)
    mse_output = ((output_img_cv2 - img_high) ** 2).mean()#108.776541 108.6
    print(mse_output)
    cv2.imshow('Image', output_img_cv2)
    cv2.waitKey(0)  # 等待按键事件
    cv2.destroyAllWindows()

