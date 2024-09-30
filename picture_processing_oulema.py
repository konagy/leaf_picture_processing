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
    cv2.drawContours(leaf1, [approx_contour1], -1, color=0, thickness=6)

    leaf2 = cv2.bitwise_and(image, image, mask=mask2)
    cv2.drawContours(leaf2, [approx_contour2], -1, color=0, thickness=6)

    # Place leaf on white background by copying only the non-black pixels
    leaf1_on_white = np.where(leaf1 == 0, white_bg1, leaf1)
    leaf2_on_white = np.where(leaf2 == 0, white_bg2, leaf2)

    if True:
        cv2.imwrite(result_path + "/" + filename_ID + '_leaf1.jpg', leaf1_on_white)
        cv2.imwrite(result_path + "/" + filename_ID + '_leaf2.jpg', leaf2_on_white)

    del image, mask, hsv_image, contours, mask1, mask2,leaf1_on_white,leaf2_on_white   
    cv2.destroyAllWindows()
    gc.collect()

    processed_files = [result_path + "/" + filename_ID + '_leaf1.jpg', result_path + "/" + filename_ID + '_leaf2.jpg']
    return processed_files

#################################################################
# Convert image and find contours ###############################
#################################################################

def picture_processing(image_path,threshold,plotting):
    
    # Read image
    image = cv2.imread(image_path)

    folder_path, image_name = os.path.split(image_path)
    # Get name of each file
    filename_ID = image_name.rsplit('.', 1)[0]
    
    # Processing cropped image ####################################################
    gray_image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret3,white_areas_cv = cv2.threshold(gray_image_cv,threshold,250,cv2.THRESH_BINARY)
    output_image_cv = cv2.bitwise_and(image, image, mask=white_areas_cv)

    
    if True:
        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title('Determined Spots')
        plt.imshow(cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB))
        #plt.show()  
        plt.savefig(result_path + "/" + filename_ID + '_processing.jpg')
        plt.close

    # Calculate areas
    non_black_pixels = np.any(output_image_cv != [0, 0, 0], axis=-1)
    spotArea = np.count_nonzero(non_black_pixels)
    non_black_pixels = np.any(image != [0, 0, 0], axis=-1)
    allArea = np.count_nonzero(non_black_pixels)
    
    # calculate ratio of leaf/spots
    try:
        spotPercentage = (spotArea / allArea) * 100
    except:
        print("ERROR - something got wrong")
        allArea = 999
        spotPercentage = 999
        pass

    ##################### Generate output #####################
    cv2.imwrite(result_path + "/" + filename_ID + "-green_object" + ".jpg", output_image_cv)

    append_to_csv(result_file_name, (image_name, allArea, spotArea, spotPercentage))

    # Print for debugging
    if plotting:
        print(f"All area = {allArea}")
        print(f"Spot area = {spotArea}")
        print(f"Percentage = {spotPercentage}")

    del image, output_image_cv, gray_image_cv, white_areas_cv
    cv2.destroyAllWindows()
    gc.collect()
    return

#################################################################
# GUI feature ###################################################
#################################################################
def threshold_finder_GUI(image_path):

    # Load the image
    image_cv = cv2.imread(image_path)
    scale_percent = 20  # Percentage of original size
    width = int(image_cv.shape[1] * scale_percent / 100)
    height = int(image_cv.shape[0] * scale_percent / 100)
    image_cv = cv2.resize(image_cv, (width, height))

    threshold = 0
    val2 = 0

    # Callback function for trackbars (does nothing but required for trackbars)
    def nothing(x):
        pass

    def real_time_image_processing():
        
        cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trackbars', 400, 200)  # Resize the trackbar window
        cv2.createTrackbar('threshold', 'Trackbars', 115, 255, nothing)  
        cv2.createTrackbar('val2', 'Trackbars', 255, 255, nothing)

        while True:
            
            
            # Get current positions of the trackbars
            threshold = cv2.getTrackbarPos('threshold', 'Trackbars')
            val2 = cv2.getTrackbarPos('val2', 'Trackbars')
            # Convert the image to grayscale
            gray_image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            # Define a threshold to identify white/pale areas (values above 200)
            #_, white_areas_cv = cv2.threshold(gray_image_cv, threshold, val2, cv2.THRESH_BINARY)
            

            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gray_image_cv,(5,5),0)
            ret3,white_areas_cv = cv2.threshold(gray_image_cv,threshold,val2,cv2.THRESH_BINARY)
            #ret3,white_areas_cv = cv2.threshold(gray_image_cv,threshold,val2,cv2.THRESH_BINARY)
            
            
            #cv2.imshow("asd",gray_image_cv)
            #cv2.waitKey(0)
            
            # Create a mask where the white areas will be highlighted
            output_image_cv = cv2.bitwise_and(image_cv, image_cv, mask=white_areas_cv)
            non_black_pixels = np.any(output_image_cv != [0, 0, 0], axis=-1)
            spotArea = np.count_nonzero(non_black_pixels)
            print("spots area = ",spotArea)

            non_black_pixels = np.any(image_cv != [0, 0, 0], axis=-1)
            allArea = np.count_nonzero(non_black_pixels)
            print("picture area",allArea)
            percentage = (spotArea/allArea)*100
            cv2.putText(output_image_cv, "Spot area = " + str(percentage) + "%", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            cv2.imshow("Cropped Image", output_image_cv)
            #cv2.waitKey(0)
            # Convert from BGR to RGB for displaying via matplotlib
            output_image_rgb = cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB)
            #cv2.imshow('Processed Image', output_image_rgb)
            
            time.sleep(0.25)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Cleanup
        cv2.destroyAllWindows()

    # Call the function
    real_time_image_processing(threshold)


################################
########## Inputs ##############
################################

folder_path = './Oulema'
plotting = True
spot_threshold = 115
test_file_index = 0
GUI_enabled = False

################################
########## Main code ###########
################################

#folder_path = './results-2024-09-30-14-02-01'
files = os.listdir(folder_path)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

if GUI_enabled:
    processed_files = splitting_images(folder_path + "\\" +  files[test_file_index])
    spot_threshold = threshold_finder_GUI(processed_files[0])

# Print the file names
for file in files:
    try:
        print("Processing: " + file)
        processed_files = splitting_images(folder_path + "\\" +  file)
        for file in processed_files:
            picture_processing(file, spot_threshold, plotting)
        print("##########################################")
    except Exception as error:
        # handle the exception
        print("An error occurred:", error)
        append_to_csv(result_file_name, ("An error occurred:", error, file))

print("Processing done!")