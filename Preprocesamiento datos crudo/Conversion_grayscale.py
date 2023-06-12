

## CONVERSIÓN DE LOS DATASETS A ESCALA DE GRISES
# AUTOR: Nuria Velasco Pérez

from PIL import Image  # Importar la clase Image del módulo PIL
import os  # Importar el módulo os para trabajar con archivos y directorios

# Directorios de procesamiento y no procesamiento de imágenes
dir_proc = 'Datos cross validation EMD'
dir_proc_grayscale = 'Datos cross validation EMD grayscale'
dir_no_proc = 'Datos cross validation base EMD'
dir_no_proc_grayscale = 'Datos cross validation base EMD grayscale'

# Bucle para recorrer los diferentes conjuntos de datos ('K1', 'K2', 'K3', 'K4', 'K5')
for e in ['K1', 'K2', 'K3', 'K4', 'K5']:
    # Bucle para recorrer los diferentes tipos de imágenes ('Samsung', 'iPhone', 'OCT', 'MESSIDOR')
    for i in ['Samsung', 'iPhone', 'OCT', 'MESSIDOR']:
        # Bucle para recorrer las diferentes opciones de procesamiento ('EMD', 'NO EMD')
        for k in ['EMD', 'NO EMD']:
            # Bucle para recorrer las diferentes opciones de procesamiento en escala de grises ('EMD', 'NO EMD')
            for w in ['EMD', 'NO EMD']:
                # Obtener la lista de archivos en el directorio correspondiente
                lista = os.listdir(dir_no_proc+'/'+e+'/'+i+'/'+k+'/'+w)
                
                # Bucle para procesar cada imagen de la lista
                for x in lista:
                    # Abrir la imagen original utilizando la clase Image
                    imagen = Image.open(dir_proc+'/'+e+'/'+i+'/'+k+'/'+w+'/'+x)
                    
                    # Convertir la imagen a escala de grises
                    imagen_gris = imagen.convert("L")
                    
                    # Guardar la imagen en escala de grises en el directorio correspondiente con un nuevo nombre
                    imagen_gris.save(dir_no_proc_grayscale+'/'+e+'/'+i+'/'+k+'/'+w+'/'+x+'_gray.jpg')

