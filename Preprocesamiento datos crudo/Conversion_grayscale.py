#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import os

dir_proc = 'Datos cross validation EMD'
dir_proc_grayscale = 'Datos cross validation EMD grayscale'
dir_no_proc = 'Datos cross validation base EMD'
dir_no_proc_grayscale = 'Datos cross validation base EMD grayscale'

for e in ['K1', 'K2', 'K3', 'K4', 'K5']:
    for i in ['Samsung', 'iPhone', 'OCT', 'MESSIDOR']:
        for k in ['EMD', 'NO EMD']:
            for w in ['EMD', 'NO EMD']:
                lista = os.listdir(dir_no_proc+'/'+e+'/'+i+'/'+k+'/'+w)
                for x in lista:
                    imagen = Image.open(dir_proc+'/'+e+'/'+i+'/'+k+'/'+w+'/'+x)
                    imagen_gris = imagen.convert("L")
                    imagen_gris.save(dir_no_proc_grayscale+'/'+e+'/'+i+'/'+k+'/'+w+'/'+x+'_gray.jpg')

