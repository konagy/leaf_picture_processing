import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
import time
import os
import gc

# Function to initialize and write the first row
def initialize_csv(file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header and the first row
        writer.writerow(["Image name", "Area of leaf [pixels]", "Area of spots [pixels]", "Percentage of spots [%]", "Number of spots [pcs]", "Number of spots [pcs]"])

# Function to write additional rows
def append_to_csv(file_path, data=[]):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write additional rows
        writer.writerow(data)

# Get time
t = time.localtime()
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", t)
print("Current time: ", current_time)

# Set results directories ################################

result_path = ('results-' +  str(current_time))
# Define the root folder and the results folder
root_folder = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(root_folder, result_path)
# create Results folder if not existing
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

result_file_name = result_path + "/results-" + str(current_time) + ".csv"
# Define the path to the CSV file
csv_file_path = os.path.join(results_folder, result_file_name)
# Create csv file with header
initialize_csv(result_file_name)


# Callback function for trackbars (does nothing but required for trackbars)
def nothing(x):
    pass

def real_time_image_processing(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Resize the image to fit smaller windows (adjust this as necessary)
    scale_percent = 20  # Percentage of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height))
    epsilon_factor = 0



    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a window for trackbars
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 200)  # Resize the trackbar window

    # Create trackbars for Hue, Saturation, and Value
    cv2.createTrackbar('LH', 'Trackbars', 0, 179, nothing)  # Lower Hue
    cv2.createTrackbar('LS', 'Trackbars', 0, 255, nothing)  # Lower Saturation
    cv2.createTrackbar('LV', 'Trackbars', 0, 255, nothing)  # Lower Value
    cv2.createTrackbar('UH', 'Trackbars', 179, 179, nothing)  # Upper Hue
    cv2.createTrackbar('US', 'Trackbars', 255, 255, nothing)  # Upper Saturation
    cv2.createTrackbar('UV', 'Trackbars', 255, 255, nothing)  # Upper Value
    cv2.createTrackbar('epsilon', 'Trackbars', 0, 1000, nothing)  # epsilon_factor

    while True:
        # Get current positions of the trackbars
        l_h = cv2.getTrackbarPos('LH', 'Trackbars')
        l_s = cv2.getTrackbarPos('LS', 'Trackbars')
        l_v = cv2.getTrackbarPos('LV', 'Trackbars')
        u_h = cv2.getTrackbarPos('UH', 'Trackbars')
        u_s = cv2.getTrackbarPos('US', 'Trackbars')
        u_v = cv2.getTrackbarPos('UV', 'Trackbars')
        epsilon = cv2.getTrackbarPos('epsilon', 'Trackbars')
        
        epsilon_factor = epsilon/10000

        # Set lower and upper bounds for green color in HSV
        lower_green = np.array([l_h, l_s, l_v])
        upper_green = np.array([u_h, u_s, u_v])

        # Create a mask using the lower and upper bounds
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        contoursGreen, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_object = cv2.bitwise_and(image, image, mask=mask)
        #gray = cv2.cvtColor(green_object, cv2.COLOR_BGR2GRAY)
        ret2,thresh2 = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh2, 1, 2)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        def approximate_contour(contour, epsilon_factor):
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            return approx
        
        areas = []
        i = 0
        eachArea = 0
        contour_area_list = []
        spotArea = 0

        # Delete small areas #################################
        for contour in contours:
            eachArea = cv2.contourArea(contour)
            if eachArea >= 4:
                #areas.append(eachArea)
                contour_area_list.append({'contour': contour, 'area': eachArea})
                i += 1

        # Find largest area (green area) #################################
        if contour_area_list:
            max_contour = max(contour_area_list, key=lambda x: x['area'])
            largest_contour = max_contour['contour']
            largest_area = max_contour['area']
        # Draw contour around the green object
        largest_contour = approximate_contour(largest_contour,epsilon_factor)
        cv2.drawContours(green_object, largest_contour, -1, 255, 3)

        # List all small contours #################################
        spots = [item for item in contour_area_list if item['contour'] is not largest_contour]
        #numbOfSpots = len(spots)
        # Draw contours for each spot
        for item in spots:
            #item['contour'] = approximate_contour(item['contour'], epsilon_factor)
            cv2.drawContours(green_object, item['contour'], -1, (0, 255, 0), 1)
            spotArea += item['area']
            areas.append(item['area'])

        # Show the original image and the processed image
        #cv2.imshow('Original Image', image)
        cv2.imshow('Processed Image', green_object)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()

# Path to the image you want to process
image_path = '.\\pictures_rect\\PXL_20260326_100705089.jpg'

# Call the function
real_time_image_processing(image_path)
