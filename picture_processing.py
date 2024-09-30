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


#################################################################
# Separating two leaves on one picture into two objects #########
#################################################################
def splitting_images(image_path):
    image = cv2.imread(image_path)
    
    folder_path, image_name = os.path.split(image_path)
    # Get name of each file
    filename_ID = image_name.rsplit('.', 1)[0]

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([36, 25, 25])  # Lower bound of green
    upper_green = np.array([86, 255, 255])  # Upper bound of green
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]


    def approximate_contour(contour, epsilon_factor=0.001):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    # Create blank white images
    white_bg1 = np.ones_like(image) * 0  # White background for leaf 1
    white_bg2 = np.ones_like(image) * 0  # White background for leaf 2
    
    # Create masks for each leaf
    mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
    
    approx_contour1 = approximate_contour(contours[0])
    cv2.drawContours(mask1, [approx_contour1], -1, 255, thickness=cv2.FILLED)

    approx_contour2 = approximate_contour(contours[1])
    cv2.drawContours(mask2, [approx_contour2], -1, 255, thickness=cv2.FILLED)

    # Extract each leaf using the masks and combine with white background
    leaf1 = cv2.bitwise_and(image, image, mask=mask1)
    leaf2 = cv2.bitwise_and(image, image, mask=mask2)

    # Place leaf on white background by copying only the non-black pixels
    leaf1_on_white = np.where(leaf1 == 0, white_bg1, leaf1)
    leaf2_on_white = np.where(leaf2 == 0, white_bg2, leaf2)

    # Optionally display the result for verification
    if False:
        #plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(leaf1_on_white, cv2.COLOR_BGR2RGB))
        #plt.title('Leaf 1')
        plt.axis('off')
        plt.savefig(result_path + "/" + filename_ID + '_leaf1.jpg',dpi=900)
        plt.close()
        #plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(leaf2_on_white, cv2.COLOR_BGR2RGB))
        #plt.title('Leaf 2')
        plt.axis('off')
        plt.savefig(result_path + "/" + filename_ID + '_leaf2.jpg',dpi=900)
        plt.show()
        plt.close()
    if True:
        cv2.imwrite(result_path + "/" + filename_ID + '_leaf1.jpg', leaf1_on_white)
        # Show the cropped image
        cv2.imshow("Cropped Image", leaf1_on_white)
        cv2.waitKey(0)
    quit()
    del image, mask, hsv_image, contours, mask1, mask2,leaf1_on_white,leaf2_on_white   
    cv2.destroyAllWindows()
    gc.collect()

    #return leaf1,leaf2
    return

#################################################################
# Convert image and find contours ###############################
#################################################################

def picture_processing(image_path):
    
    # Read image
    image = cv2.imread(image_path)

    folder_path, image_name = os.path.split(image_path)
    # Get name of each file
    filename_ID = image_name.rsplit('.', 1)[0]
    # Find big contour and crop the image so that the black frame will be deleted
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to create a mask
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Find the contours of the white areas
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the bounding box of the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    #cropped_image_path = ".\pictures\cropped.jpg"
    #cv2.imwrite(cropped_image_path, cropped_image)
    # Show the cropped image
    #cv2.imshow("Cropped Image", cropped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    if False:
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 2, 2)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
        plt.show()    

    # Processing cropped image ####################################################

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    # Define the range for the green color in HSV
    # original idea: lower_green = np.array([35, 40, 40]) / upper_green = np.array([85, 255, 255])
    #light_green = np.array([30,100,30])
    #dark_green = np.array([230,255,230])
    light_green = np.array([20, 20, 20])
    dark_green = np.array([85, 255, 255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_image, light_green, dark_green)
    # Find contours of the green area
    contoursGreen, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the area of the green object in pixels // we'll calculate it again later
    #green_area = sum(cv2.contourArea(contour) for contour in contoursGreen) 

    # Processing the green object ####################################################
    # Isolate the green object from the original image using the mask
    green_object = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    gray = cv2.cvtColor(green_object, cv2.COLOR_BGR2GRAY)
    ret2,thresh2 = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh2, 1, 2)

    #cv2.imshow("green_object", green_object)
    #cv2.waitKey(0)
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.imshow("thresh2", thresh2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if False: 
        # Plotting the original and masked image for visualization
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 2, 2)
        plt.title('Green Object')
        plt.imshow(mask, cmap='gray')
        plt.subplot(2, 2, 3)
        plt.title('Green Object With Mask')
        plt.imshow(cv2.cvtColor(green_object, cv2.COLOR_BGR2RGB))
        plt.savefig(result_path + "/" + filename_ID + '_processing.jpg')
        plt.show()
        plt.close()



    # Calculating areas adn drawing contours on output #######################################################
    # Calculate the area of the white spot in pixels

    areas = []
    i = 0
    eachArea = 0
    contour_area_list = []
    spotArea = 0

    # Create a list for each contours that is bigger then 4 pixels
    for contour in contours:
        eachArea = cv2.contourArea(contour)
        if eachArea >= 4:
            #areas.append(eachArea)
            contour_area_list.append({'contour': contour, 'area': eachArea})
            i += 1

    # Find largest area (green area).
    if contour_area_list:
        max_contour = max(contour_area_list, key=lambda x: x['area'])
        largest_contour = max_contour['contour']
        largest_area = max_contour['area']

    # Draw contour around the green object
    cv2.drawContours(green_object, largest_contour, -1, 255, 3)

    # List all small contours
    spots = [item for item in contour_area_list if item['contour'] is not largest_contour]
    numbOfSpots = len(spots)

    # Draw contours for each spot
    for item in spots:
        cv2.drawContours(green_object, item['contour'], -1, (0, 255, 0), 3)
        spotArea += item['area']
        areas.append(item['area'])

    # calculate ratio of leaf/spots
    try:
        spotPercentage = (spotArea / largest_area) * 100
    except:
        print("ERROR - something got wrong")
        largest_area = 999
        spotPercentage = 999
        pass

    ##################### Generate output #####################
    cv2.imwrite(result_path + "/" + filename_ID + "-green_object" + ".jpg", green_object)

    append_to_csv(result_file_name, (image_name, largest_area, spotArea, spotPercentage, numbOfSpots, areas))

    # Print for debugging
    if True:
        print("Number of small spots:", numbOfSpots)
        print(areas)
        print(f"All area = {largest_area}")
        print(f"Spot area = {spotArea}")
        print(f"Percentage = {spotPercentage}")

    return


# Load the image 
#image_folder = '.\pictures\\'
#image_name = 'IMG_20200928_082250.jpg'
#image_path = image_folder + image_name
#picture_processing(image_path)

# Define the folder path
#folder_path = './Oulema'
folder_path = './Oulema-test'
files = os.listdir(folder_path)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Print the file names
for file in files:
    try:
        print("Processing: " + file)
        splitting_images(folder_path + "\\" +  file)
        #picture_processing(folder_path + "\\" +  file)
        print("##########################################")
    except Exception as error:
        # handle the exception
        print("An error occurred:", error)
        append_to_csv(result_file_name, ("An error occurred:", error, file))