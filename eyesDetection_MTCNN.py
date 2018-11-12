#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author  thiago@fatecourinhos.edu.br
# @since   2018-11
# @version v1.0
#
# PID - Prof. Ap. Nilceu Marana
# PPGCC Unesp SJRP/Bauru
#
# Script para deteccao de Olhos na base ARFACE
# utilizando MTCNN
#
# Escrito e testado no Linux Mint (Kernel 4.15.0)

print("--- eyesDetection_MTCNN.py ---")
print("-> Iniciando...")
from mtcnn.mtcnn import MTCNN
import cv2
print("-> Libs importadas...")

fnMtcnn = MTCNN()

print("-> Listando imagens...")
dir='.' # Testado sendo executado no mesmo diretorio das imagens da base ARFACE
img_regex = 'm-' # Somentes imagens *m-*
imagens_encontradas = [f for f in os.listdir(dir) if img_regex in f]
print("-> Lista de imagens ok...")

for img in imagens_encontradas:
    print("-> Processando " + img)
    try:
        image = cv2.imread(img)
        result = fnMtcnn.detect_faces(image)
        #print(result)
        keypoints = result[0]['keypoints']

        left = image[keypoints['left_eye'][1]-35:keypoints['left_eye'][1]+35,keypoints['left_eye'][0]-35:keypoints['left_eye'][0]+35]
        cv2.namedWindow("left")
        #cv2.imshow("left",left)
        newimg = "r_" + img
        cv2.imwrite(newimg, left)

        right = image[keypoints['right_eye'][1]-35:keypoints['right_eye'][1]+35,keypoints['right_eye'][0]-35:keypoints['right_eye'][0]+35]
        cv2.namedWindow("right")
        #cv2.imshow("right",right)
        newimg = "l_" + img
        cv2.imwrite(newimg, right)

        #cv2.waitKey(0)

    except:
        print("Erro na imagem " + img)
        newimg = "ERROR_" + img # a imagem com erro sera copiada com nome ERROR_[...].bmp
        cv2.imwrite(newimg, image)
