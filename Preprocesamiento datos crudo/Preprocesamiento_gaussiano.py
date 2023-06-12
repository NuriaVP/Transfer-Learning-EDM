## PREPROCESAMIENTO GAUSSIANO DE TODOS LOS SETS DE IMÁGENES

## AUTOR: Nuria Velasco Pérez

import os
import cv2
import PIL
import os
from PIL import Image
import numpy as np


# **PREPROCESADO GAUSSIANO**

#Recortar una imagen en escala de grises eliminando los bordes oscuros.

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:  # Verificar si la imagen es de 2 dimensiones (escala de grises)
        mask = img > tol  # Crear una máscara booleana donde los píxeles más brillantes que 'tol' se marcan como True
        return img[np.ix_(mask.any(1), mask.any(0))]  # Recortar la imagen usando la máscara
    elif img.ndim == 3:  # Verificar si la imagen es de 3 dimensiones (imagen RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convertir la imagen a escala de grises
        mask = gray_img > tol  # Crear una máscara booleana en escala de grises
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]  # Verificar la forma de los canales de color después del recorte
        if check_shape == 0:
            return img  # Si no hay píxeles válidos después del recorte, devolver la imagen original
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]  # Recortar el canal rojo de la imagen
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]  # Recortar el canal verde de la imagen
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]  # Recortar el canal azul de la imagen
            img = np.stack([img1, img2, img3], axis=-1)  # Combinar los canales recortados nuevamente en una imagen RGB
        return img

#Crear un recorte circular alrededor del centro de la imagen.

def circle_crop(img, sigmaX):   
    img = crop_image_from_gray(img)  # Recortar la imagen eliminando los bordes oscuros
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir la imagen a formato RGB
    
    height, width, depth = img.shape  # Obtener las dimensiones de la imagen
    
    x = int(width / 2)  # Coordenada x del centro de la imagen
    y = int(height / 2)  # Coordenada y del centro de la imagen
    r = np.amin((x, y))  # Radio del círculo
    
    circle_img = np.zeros((height, width), np.uint8)  # Crear una imagen en blanco del mismo tamaño que la imagen original
    cv2.circle(circle_img, (x, y), int(r), 1


# **CONVERTIR TODAS LAS IMÁGENES EN FORMATO JPG**


for e in os.listdir('Datos EMD/iPhone/EMD'):
    im = Image.open('Datos EMD/iPhone/EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/iPhone/EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/iPhone/EMD/'+e)


for e in os.listdir('Datos EMD/iPhone/NO EMD'):
    im = Image.open('Datos EMD/iPhone/NO EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/iPhone/NO EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/iPhone/NO EMD/'+e)


for e in os.listdir('Datos EMD/Samsung/EMD'):
    im = Image.open('Datos EMD/Samsung/EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/Samsung/EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/Samsung/EMD/'+e)


for e in os.listdir('Datos EMD/Samsung/NO EMD'):
    im = Image.open('Datos EMD/Samsung/NO EMD/'+e)
    rgb_im = im.convert('RGB')
    rgb_im.save('Datos EMD/Samsung/NO EMD/'+e[0:-3]+'jpg', quality=95)
    os.remove('Datos EMD/Samsung/NO EMD/'+e)


# **OBTENER CARPETA CON IMÁGENES PREPROCESADAS**


for e in os.listdir('Datos EMD/iPhone/EMD'):
    img = cv2.imread('Datos EMD/iPhone/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/iPhone/EMD/'+e[0:-4]+'_proc.jpg')

for e in os.listdir('Datos EMD/iPhone/NO EMD'):
    img = cv2.imread('Datos EMD/iPhone/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/iPhone/NO EMD/'+e[0:-4]+'_proc.jpg')


for e in os.listdir('Datos EMD/Samsung/EMD'):
    img = cv2.imread('Datos EMD/Samsung/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/Samsung/EMD/'+e[0:-4]+'_proc.jpg')


for e in os.listdir('Datos EMD/Samsung/NO EMD'):
    img = cv2.imread('Datos EMD/Samsung/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/Samsung/NO EMD/'+e[0:-4]+'_proc.jpg')


for e in os.listdir('Datos EMD/OCT/EMD'):
    img = cv2.imread('Datos EMD/OCT/EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/OCT/EMD/'+e[0:-4]+'_proc.jpg')

for e in os.listdir('Datos EMD/OCT/NO EMD'):
    img = cv2.imread('Datos EMD/OCT/NO EMD/'+e)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = circle_crop(img, 30)
    image = Image.fromarray(img_t, 'RGB')
    image.save('Datos preprocesados EMD/OCT/NO EMD/'+e[0:-4]+'_proc.jpg')



