#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import PIL
import os
from PIL import Image
import numpy as np


# **PREPROCESADO GAUSSIANO**

# In[2]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


# In[3]:


def circle_crop(img, sigmaX):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 


# **CONVERTIR TODAS LAS IMÁGENES EN FORMATO JPG**

# In[11]:


for e in os.listdir('Datos EMD/iPhone/EMD'):
    im = Image.open('Datos EMD/iPhone/EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/iPhone/EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/iPhone/EMD/'+e)


# In[12]:


for e in os.listdir('Datos EMD/iPhone/NO EMD'):
    im = Image.open('Datos EMD/iPhone/NO EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/iPhone/NO EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/iPhone/NO EMD/'+e)


# In[13]:


for e in os.listdir('Datos EMD/Samsung/EMD'):
    im = Image.open('Datos EMD/Samsung/EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/Samsung/EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/Samsung/EMD/'+e)


# In[14]:


for e in os.listdir('Datos EMD/Samsung/NO EMD'):
    im = Image.open('Datos EMD/Samsung/NO EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/Samsung/NO EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/Samsung/NO EMD/'+e)


# **OBTENER CARPETA CON IMÁGENES PREPROCESADAS**

# In[15]:


for e in os.listdir('Datos EMD/iPhone/EMD'):
    img = cv2.imread('Datos EMD/iPhone/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/iPhone/EMD/'+e[0:-4]+'_proc.jpg')


# In[16]:


for e in os.listdir('Datos EMD/iPhone/NO EMD'):
    img = cv2.imread('Datos EMD/iPhone/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/iPhone/NO EMD/'+e[0:-4]+'_proc.jpg')


# In[17]:


for e in os.listdir('Datos EMD/Samsung/EMD'):
    img = cv2.imread('Datos EMD/Samsung/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/Samsung/EMD/'+e[0:-4]+'_proc.jpg')


# In[18]:


for e in os.listdir('Datos EMD/Samsung/NO EMD'):
    img = cv2.imread('Datos EMD/Samsung/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/Samsung/NO EMD/'+e[0:-4]+'_proc.jpg')


# In[19]:


for e in os.listdir('Datos EMD/OCT/EMD'):
    img = cv2.imread('Datos EMD/OCT/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/OCT/EMD/'+e[0:-4]+'_proc.jpg')


# In[20]:


for e in os.listdir('Datos EMD/OCT/NO EMD'):
    img = cv2.imread('Datos EMD/OCT/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/OCT/NO EMD/'+e[0:-4]+'_proc.jpg')


# **PREPROCESAMIENTO NUEVAS IMÁGENES INP**

# In[4]:


for e in os.listdir('Datos INP EMD/iPhone/EMD'):
    img = cv2.imread('Datos INP EMD/iPhone/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/iPhone/EMD/'+e[0:-4]+'_proc.jpg')


# In[5]:


for e in os.listdir('Datos INP EMD/iPhone/NO EMD'):
    img = cv2.imread('Datos INP EMD/iPhone/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/iPhone/NO EMD/'+e[0:-4]+'_proc.jpg')


# In[6]:


for e in os.listdir('Datos INP EMD/Samsung/EMD'):
    img = cv2.imread('Datos INP EMD/Samsung/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/Samsung/EMD/'+e[0:-4]+'_proc.jpg')


# In[7]:


for e in os.listdir('Datos INP EMD/Samsung/NO EMD'):
    img = cv2.imread('Datos INP EMD/Samsung/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/Samsung/NO EMD/'+e[0:-4]+'_proc.jpg')


# In[8]:


for e in os.listdir('Datos INP EMD/OCT/EMD'):
    img = cv2.imread('Datos INP EMD/OCT/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/OCT/EMD/'+e[0:-4]+'_proc.jpg')


# In[9]:


for e in os.listdir('Datos INP EMD/OCT/NO EMD'):
    img = cv2.imread('Datos INP EMD/OCT/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/OCT/NO EMD/'+e[0:-4]+'_proc.jpg')


# **PREPROCESAMIENTO IMÁGENES MESSIDOR**

# In[4]:


for e in os.listdir('Datos INP EMD/MESSIDOR/EMD'):
    im = Image.open('Datos INP EMD/MESSIDOR/EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos INP EMD/MESSIDOR/EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos INP EMD/MESSIDOR/EMD/'+e)


# In[5]:


for e in os.listdir('Datos INP EMD/MESSIDOR/NO EMD'):
    im = Image.open('Datos INP EMD/MESSIDOR/NO EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos INP EMD/MESSIDOR/NO EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos INP EMD/MESSIDOR/NO EMD/'+e)


# In[6]:


for e in os.listdir('Datos INP EMD/MESSIDOR/EMD'):
    img = cv2.imread('Datos INP EMD/MESSIDOR/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/MESSIDOR/EMD/'+e[0:-4]+'_proc.jpg')


# In[7]:


for e in os.listdir('Datos INP EMD/MESSIDOR/NO EMD'):
    img = cv2.imread('Datos INP EMD/MESSIDOR/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados INP EMD/MESSIDOR/NO EMD/'+e[0:-4]+'_proc.jpg')

