#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import os
import shutil


# ## Base 11

# In[26]:


base11 = pd.read_excel("Annotation_Base11.xls")

display(base11)


# In[27]:


lista_edm = []
lista_no_edm = []

for e in range(len(base11)):
    
    name = base11.iloc[e][0]
    edm = base11.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)


# In[28]:


print(lista_edm)
print(lista_no_edm)


# In[29]:


imagenes_base11 = os.listdir('Base11')

print(imagenes_base11)


# In[30]:


for e in imagenes_base11:
    if e in lista_edm:
        shutil.copy("Base11/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base11/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 12

# In[31]:


base12 = pd.read_excel("Annotation_Base12.xls")

display(base12)


# In[32]:


lista_edm = []
lista_no_edm = []

for e in range(len(base12)):
    
    name = base12.iloc[e][0]
    edm = base12.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)


# In[33]:


imagenes_base12 = os.listdir('Base12')

print(imagenes_base12)


# In[34]:


for e in imagenes_base12:
    if e in lista_edm:
        shutil.copy("Base12/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base12/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 13

# In[35]:


base13 = pd.read_excel("Annotation_Base13.xls")

display(base13)


# In[36]:


lista_edm = []
lista_no_edm = []

for e in range(len(base13)):
    
    name = base13.iloc[e][0]
    edm = base13.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)


# In[37]:


imagenes_base13 = os.listdir('Base13')

print(imagenes_base13)


# In[38]:


for e in imagenes_base13:
    if e in lista_edm:
        shutil.copy("Base13/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base13/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 14

# In[39]:


base14 = pd.read_excel("Annotation_Base14.xls")

display(base14)


# In[40]:


lista_edm = []
lista_no_edm = []

for e in range(len(base14)):
    
    name = base14.iloc[e][0]
    edm = base14.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)


# In[41]:


imagenes_base14 = os.listdir('Base14')

print(imagenes_base14)


# In[42]:


for e in imagenes_base14:
    if e in lista_edm:
        shutil.copy("Base14/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base14/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 21

# In[43]:


base21 = pd.read_excel("Annotation_Base21.xls")

display(base21)


# In[44]:


lista_edm = []
lista_no_edm = []

for e in range(len(base21)):
    
    name = base21.iloc[e][0]
    edm = base21.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)


# In[45]:


imagenes_base21 = os.listdir('Base21')

print(imagenes_base21)


# In[46]:


for e in imagenes_base21:
    if e in lista_edm:
        shutil.copy("Base21/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base21/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 22

# In[47]:


base22 = pd.read_excel("Annotation_Base22.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base22)):
    
    name = base22.iloc[e][0]
    edm = base22.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base22 = os.listdir('Base22')


# In[48]:


for e in imagenes_base22:
    if e in lista_edm:
        shutil.copy("Base22/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base22/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 23

# In[49]:


base23 = pd.read_excel("Annotation_Base23.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base23)):
    
    name = base23.iloc[e][0]
    edm = base23.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base23 = os.listdir('Base23')


# In[50]:


for e in imagenes_base23:
    if e in lista_edm:
        shutil.copy("Base23/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base23/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 24

# In[51]:


base24 = pd.read_excel("Annotation_Base24.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base24)):
    
    name = base24.iloc[e][0]
    edm = base24.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base24 = os.listdir('Base24')


# In[52]:


for e in imagenes_base24:
    if e in lista_edm:
        shutil.copy("Base24/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base24/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 31

# In[53]:


base31 = pd.read_excel("Annotation_Base31.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base31)):
    
    name = base31.iloc[e][0]
    edm = base31.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base31 = os.listdir('Base31')


# In[54]:


for e in imagenes_base31:
    if e in lista_edm:
        shutil.copy("Base31/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base31/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 32

# In[55]:


base32 = pd.read_excel("Annotation_Base32.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base32)):
    
    name = base32.iloc[e][0]
    edm = base32.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base32 = os.listdir('Base32')


# In[56]:


for e in imagenes_base32:
    if e in lista_edm:
        shutil.copy("Base32/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base32/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 33

# In[57]:


base33 = pd.read_excel("Annotation_Base33.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base33)):
    
    name = base33.iloc[e][0]
    edm = base33.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base33 = os.listdir('Base33')


# In[58]:


for e in imagenes_base33:
    if e in lista_edm:
        shutil.copy("Base33/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base33/"+e, "MESSIDOR/NO EMD/"+e)


# ## Base 34

# In[59]:


base34 = pd.read_excel("Annotation_Base34.xls")

lista_edm = []
lista_no_edm = []

for e in range(len(base34)):
    
    name = base34.iloc[e][0]
    edm = base34.iloc[e][2]
    
    if (edm == 0):
        lista_edm.append(name)
    else:
        lista_no_edm.append(name)

imagenes_base34 = os.listdir('Base34')


# In[60]:


for e in imagenes_base34:
    if e in lista_edm:
        shutil.copy("Base34/"+e, "MESSIDOR/EMD/"+e)
    else:
        shutil.copy("Base34/"+e, "MESSIDOR/NO EMD/"+e)

