import cv2
import numpy
import numpy as np
import os
#调整输入输出图片的尺寸使它们一致
file_pathname="data/for1000/val/lowl"
for filename in os.listdir(file_pathname):
    print(filename)
    img = cv2.imread(file_pathname + "/" + filename, 1)
    new_dimensions = (400, 400)
    resized_image=cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)
    savefile = "data/for1000/val/low/" + filename
    cv2.imwrite(savefile, resized_image)