import glob
import os

import numpy as np
from PIL import Image, ImageDraw
import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

def blood_add(img):
    # read input image
    
    height, width = img.shape[:2]

    # define random seed to change the pattern
    # seedval = 75
    rng = default_rng()

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 120, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.merge([mask,mask,mask])
    binary_mask = np.array(mask, dtype=bool)
    binary_img = np.array(img, dtype=bool)
    comb_img= cv2.merge([img,img,img])

    # add mask to input
    comb_img[:, :, :3][binary_mask] = [180,0,0]
    comb_img[np.invert(binary_img)] =[0,0,0]
    # res2=mask[:,:,:3][]
    # result1 = cv2.add(img, mask[:,:,0])
    # img[mask[:,:,0]]=125

    comb_img[200 <comb_img[:,:,0] ] = [0,0,0]
    comb_img[180 ==comb_img[:,:,0] ] = [255,255,255]
    return np.array(comb_img[:,:,0], dtype=bool)
    

if __name__ == '__main__':
    count_V=0
    count_X=0
    mask_srcs=sorted(glob.glob("/mask/*"))
    image_srcs = sorted(glob.glob("/source/*"))
    ds_type=["blood_white_","white_gloves_","blood_blue_","blue_gloves_"]
    pbar = tqdm(total=len(image_srcs))
    for mask_src, pic_src in zip(mask_srcs,image_srcs ):
        pbar.update(1)
        try:     
            img = np.array(Image.open(pic_src))
            blue_glove= img.copy()
            white_glove= img.copy()
            mask = np.array(Image.open(mask_src))

            blood_binary_mask= blood_add(mask)

            binary_mask = np.array(mask, dtype=bool)
            blue_glove[:, :, :3][binary_mask] =((img[:, :, :3][binary_mask]+[0, 140, 240])/2+[0, 80, 220])/2.2
            white_glove[:, :, :3][binary_mask] =((img[:, :, :3][binary_mask]+[255, 255, 255])/2+[255, 255, 255])/2
            
            blood_white_glove= white_glove.copy()
            blood_blue_glove= blue_glove.copy()
            blood_white_glove[:, :, :3][blood_binary_mask]=[100,24,24]
            blood_blue_glove[:, :, :3][blood_binary_mask]=[100,24,24]
            count_V+=1
            for img_to_save ,label in zip([white_glove,blue_glove,blood_white_glove,blood_blue_glove],["white_gloves","blue_gloves","blood_white_gloves","blood_blue_gloves"]):        
                img_to_save = Image.fromarray(img_to_save)
                src= os.path.basename(pic_src)[:-4]+'.png'
                img_to_save.save(f"HandsDetection/images/{label}/Test/{src}")
                    
        except Exception as e:
            print(e)
            count_X+=1
    print(f"number of proccesed pictures {count_V}")
    print(f"number of failed pictures {count_X}")