# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:24:09 2023

@author: Thomas
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#leitura da imagem
img = cv2.imread('img.jpeg',0)

#filtro gaussiano (pode não funcionar, talvez se diminuir o kernel possa funcionar melhor
#                  ou se retirar o filtro)
blur = cv2.GaussianBlur(img,(5,5),0)

#mask (corta da imagem bordas desnecessárias, caso imagem esteja já cortada, 
#      apenas comente o código, caso queira, é possível mudar o tamanho da 
#      borda a ser cortada também alterando a variável borda para o número 
#      de pixels ou fração da imagem)
width = img.shape[1]
height = img.shape[0]
borda = round(height/15)+15 
mask = img[borda : height-borda, 0 : width]

#binarização por método de otsu (é possível equalizar a imagem para melhorar sua qualidade
#                                caso o método não corresponda ao esperado)
ret, binary = cv2.threshold(mask,0,255,cv2.THRESH_OTSU)
#ret,binary = cv2.threshold(binary,127,255,cv2.THRESH_BINARY_INV) #caso queira inverter a imagem para melhor análise
#cv2.imshow("binario", binary) #caso queira mostrar a imagem separado só descomentar

#operações morfológicas (talvez não precise ou necessite da função open, que pode ser usada alterando para MORPH_OPEN)
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)

#contornos (observa a quantidade de trilhas encontradas na imagem)
contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#--------------analise geral-----------#
erro = 0
if (len(contours) != 100): #possível alterar o numero para o desejado de trilhas, 
#                           (caso menor ocorre um curto, caso maior, uma falha na trilha)
	erro = 1
#--------------analise individual-----------#
if erro == 1: #criação de máscara para analíse detalhada de parte da imagem 
#             (é possível criar mais máscaras e analisar a foto por setores adicionando variáveis como essa)
    mask1 = closing[0 : round(closing.shape[0]/3) , 0 : round(width/5)] 

    fotos = [mask1] #adicionar as outras fotos do passo acima aqui
    erros = 0
    for i in fotos:
        contours, hierarchy = cv2.findContours( i ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 4: #(trilhas esperadas por foto, 
#                                caso sejam peças iguais na foto completa,
#                                é esperado que sejam o mesmo número de trilhas em todas as fotos)
            erros += 1         #contagem de erros, pode ser alterada para mostrar a foto que encontrou o erro
    

plt.subplot(131),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(mask, cmap='gray'),plt.title('Mask')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(binary, cmap='gray'),plt.title('Binarização')
plt.xticks([]), plt.yticks([])
plt.show()