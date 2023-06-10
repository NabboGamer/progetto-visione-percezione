import numpy as np
import cv2

def segmentation_and_cropping(image, full_mask):
    # creo una ROI a partire dalla maschera 
    x,y,w,h = cv2.boundingRect(full_mask) 
    rectangle_image = image.copy()
    # margine extra a sinistra e destra
    extra_margin = 100 
    # no cropping su altezza
    y = 0 
    h = image.shape[0]
    # creo un rettangolo a partire dalla ROI
    cv2.rectangle(rectangle_image, (x,y), (x+w,y+h), (255,0,0), 0)
    # croppo l'immagine usando la ROI
    if (x-extra_margin) < 0 and (x+w+extra_margin) < image.shape[1]:
        cropped_image = rectangle_image[y:y+h, 0:(x+w+extra_margin)]
    elif (x-extra_margin) > 0 and (x+w+extra_margin) > image.shape[1]:
        cropped_image = rectangle_image[y:y+h, (x-extra_margin):image.shape[1]]
    elif (x-extra_margin) < 0 and (x+w+extra_margin) > image.shape[1]:
        cropped_image = rectangle_image[y:y+h, 0:image.shape[1]]
    else:
        cropped_image = rectangle_image[y:y+h, (x-extra_margin):(x+w+extra_margin)]
    # riconverto l'immagine in colori umani
    return cropped_image