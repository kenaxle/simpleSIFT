import SIFT as sf
import cv2

points, image = sf.SiftPoints('lena256.jpg', True)

if not(image is None ):
    cv2.imshow("Image",image)
    cv2.waitKey(0)
