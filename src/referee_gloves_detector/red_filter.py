import numpy as np
import cv2

def red_filtering(image):
    #cv2.imshow("Original", image) 
    result = image.copy() 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([5, 255, 255])
    lower2 = np.array([174,100,100])
    upper2 = np.array([179,255,255])  
    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)  
    full_mask = lower_mask + upper_mask
    #result = cv2.bitwise_and(result, result, mask=full_mask)
    #cv2.imshow('mask', full_mask)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return full_mask