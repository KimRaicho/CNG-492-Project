import cv2
import pytesseract
import scipy.io
import numpy as np
import os



pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
   
def crop(filename):
    
    new= 'HOUGH/' + filename
    mat = scipy.io.loadmat(new)
    
    filename= filename.split('.')
    original_image_name = filename[0]
    
    reading_details = '1-IMAGES/' + original_image_name + '.bmp'
    img = cv2.imread(reading_details)
    
    imageblack = np.zeros(img.shape, dtype = "uint8")
#-----------------------------------#


    pupil = mat['circlepupil'][0]
    iris = mat['circleiris'][0]

    irisCenter = (iris[1], iris[0]) # Iris Center coordinates
    irisRaduis = iris[2] # Reading Iris radius 

    pupilCenter = (pupil[1], pupil[0]) # Reading Pupil center coordinates 
    pupilRadius = pupil[2] # reading pupil radius 


    # drawing white circle for the iris and black circle for the pupil on the mask image 
    #------------------------------#
    thickness = -1 # to fill all the given area 

    color = (255, 255, 255) 
    imageblack = cv2.circle(imageblack, irisCenter, irisRaduis, color, thickness)

    # draw a black circle on the pupil region 
    color = (0, 0, 0)
    imageblack = cv2.circle(imageblack, pupilCenter, pupilRadius, color, thickness)

    # masking the original image with the masking image 
    im3 = cv2.bitwise_and(img, imageblack)
  
  
    return im3



# reading all the images in the HOUGH folder 
folder = "HOUGH"
for filename in os.listdir(folder):
            cropped_image = crop(filename) 
            
            # we do this because we want the original image and the output image to have almost same names 
            filename= filename.split('.') # filename[0] has the part of the name that we want. 
            saving_details = 'CroppedImages/' + filename[0] + '-cropped.jpg'
            cv2.imwrite(saving_details, cropped_image) # saving the output image to the CroppedImages folder 


