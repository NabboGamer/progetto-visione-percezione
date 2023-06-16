from Homography import Homography
import cv2

src = cv2.imread("imgs/src.jpg")
dst = cv2.imread("imgs/dst.jpg")

h = Homography(src,dst)
h._from_detection()