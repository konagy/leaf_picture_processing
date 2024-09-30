import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
# Load the image
image_path = "./results-2024-09-30-12-47-30/IMG_20210520_144119_leaf1.jpg"


# Resize the image to fit smaller windows (adjust this as necessary)

image_cv = cv2.imread(image_path)
scale_percent = 20  # Percentage of original size
width = int(image_cv.shape[1] * scale_percent / 100)
height = int(image_cv.shape[0] * scale_percent / 100)
image_cv = cv2.resize(image_cv, (width, height))

val1 = 0
val2 = 0

# Callback function for trackbars (does nothing but required for trackbars)
def nothing(x):
    pass

def real_time_image_processing():
    
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 200)  # Resize the trackbar window
    cv2.createTrackbar('val1', 'Trackbars', 115, 255, nothing)  # Lower Hue
    cv2.createTrackbar('val2', 'Trackbars', 255, 255, nothing)  # Lower Saturation

    while True:
        
        
        # Get current positions of the trackbars
        val1 = cv2.getTrackbarPos('val1', 'Trackbars')
        val2 = cv2.getTrackbarPos('val2', 'Trackbars')
        # Convert the image to grayscale
        gray_image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        # Define a threshold to identify white/pale areas (values above 200)
        #_, white_areas_cv = cv2.threshold(gray_image_cv, val1, val2, cv2.THRESH_BINARY)
        

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(gray_image_cv,(5,5),0)
        ret3,white_areas_cv = cv2.threshold(gray_image_cv,val1,val2,cv2.THRESH_BINARY)
        #ret3,white_areas_cv = cv2.threshold(gray_image_cv,val1,val2,cv2.THRESH_BINARY)
        
        
        #cv2.imshow("asd",gray_image_cv)
        #cv2.waitKey(0)
        
        # Create a mask where the white areas will be highlighted
        output_image_cv = cv2.bitwise_and(image_cv, image_cv, mask=white_areas_cv)
        non_black_pixels = np.any(output_image_cv != [0, 0, 0], axis=-1)
        count_non_black = np.count_nonzero(non_black_pixels)
        print("spots area = ",count_non_black)

        non_black_pixels = np.any(image_cv != [0, 0, 0], axis=-1)
        count_non_black = np.count_nonzero(non_black_pixels)
        print("picture area",count_non_black)

        cv2.imshow("Cropped Image", output_image_cv)
        #cv2.waitKey(0)
        # Convert from BGR to RGB for displaying via matplotlib
        output_image_rgb = cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB)
        #cv2.imshow('Processed Image', output_image_rgb)
        
        time.sleep(0.5)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()

# Call the function
real_time_image_processing()
