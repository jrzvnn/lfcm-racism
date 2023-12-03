import torch
import torchvision.transforms as transforms
from skimage import io, transform
import numpy as np 
from PIL import Image, ImageOps 
import random
from typing import Union

module_name = 'customTransform.py'
def mirror(image):
    """Mirror Image

    Args:
        image (Image): PIL.Image

    Returns:
        PIL.Image | None: returns mirrored image. None if error.
    """
    try:
        return ImageOps.mirror(image)
    except Exception as e:
        print(f"{module_name} mirror() error:",e)
        return None 

def rescale(image, output_size):
    """Rescale image to specific output size

    Args:
        image (Image): PIL.Image

        output_size (int): image dimension of (output_size x output_size)
    Returns:
        PIL.Image: returns rescaled image based on output_size. None if error.
    """
    try:
        # temporarily removed: if image.size != (output_size, output_size)
        '''
        image resizing and anti-aliasing using Lanczos algorithm since Image.ANTIALIAS
        is already deprecated
        '''
        return image.resize((output_size, output_size), Image.LANCZOS)
    except Exception as e:
        print(f"{module_name} rescale() error:",e)
        return None

def random_crop(image, output_size):
    """Random crops an image

    Args:
        image (Image): PIL.Image

        output_size (int): image dimension of (output_size x output_size)

    Returns:
        PIL.Image | None: random cropped image. None if error.
    """
    try:
        width, height = image.size
        output_image = image.copy()
        
        '''
        image resizing and anti-aliasing using Lanczos algorithm since Image.ANTIALIAS
        is already deprecated
        '''
        # resize (larger) image if (width) it is smaller than the output size
        if width < output_size - 1:
            output_image = image.resize((width * 2, height * 2), Image.LANCZOS)
        
        # resize (larger) image if (height) it is smaller than the output size
        if height < output_size - 1:
            output_image = image.resize((width * 2, height * 2), Image.LANCZOS)       
        
        #  random coordinates for random cropping
        width, height = output_image.size 
        left = random.randint(0, width - output_size - 1)
        top = random.randint(0, height - output_size - 1)

        # crop image based on generated random coordinates
        output_image = output_image.crop((left, top, left + output_size, top + output_size))
        return output_image
    except Exception as e:
        try:
            print(f"{module_name} random_crop() error:", e)
            return rescale(image, output_size)
        except:
            print(f"{module_name} random_crop() error:", e)
            return None

def center_crop(image, output_size):
    """Center crops an image

    Args:
        image (PIL.Image): PIL.Image to center crop

        output_size (int): image dimension of (output_size x output_size)

    Returns:
        PIL.Image | None: center cropped PIL.Image. None if error.
    """
    try:
        #output_image = image.copy()
        center_crop = transforms.CenterCrop(output_size)

        image_output = center_crop(image)

        return image_output 
    except Exception as e:
            print(f"{module_name} center_crop() error:", e)
            return None

def preprocess_image_to_np_arr(image, dtype=np.float32):
    """Converts image to RGB mode. Preprocesses it (mean subtraction) and returns it in ndarray.

    Args:
        image (_type_): PIL.Image to preprocess
        
        dtype (_type_, optional): numpy array data type. Defaults to np.float32.

    Returns:
        ndarray: preprocessed ndarray of image in RGB mode
    """
    try:
        #image.show()
        image_np = np.array(image, dtype=dtype)
        mean = (104.00698793, 116.66876762, 122.67891434) # mean values of RGB used for mean subtraction
        # print('initial image np:', image_np.shape)
        if (image_np.shape.__len__() < 3 or image_np.shape[2] > 3):
            im_gray = image
            image = Image.new("RGB", im_gray.size)
            image.paste(im_gray)    
            #image.show() 
            image = np.array(image, dtype=dtype)
        elif image.mode == 'RGB':
            image = image_np
        
        # reverse ndarray
        image = image[:,:,::-1]
        # centers data around zero
        image -= mean
        # transposes ndarray matrix
        image = image.transpose((2,0,1))
        return image
    except Exception as e:
        print(f"{module_name} preprocess_image_to_np_arr() error:", e)
        return None        


def filepath_to_image(filepath):
    image = None
    try:
        image = Image.open(filepath)
    except:
        None 
    return image
