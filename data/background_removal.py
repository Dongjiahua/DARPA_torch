import cv2
import numpy as np

#Background Remover with OpenCV – Method 2 – OpenCV2 Simple Thresholding
#Obviously in method 1, we performed a lot of image processing. As can be seen, Gaussian Blur, and Otsu thresholding require a lot of processing. Additionally, when applying Gaussian Blur and binning, we lost a lot of detail in our image. Hence, we wanted to design an alternative strategy that will hopefully be faster. Balanced against efficiency and knowing OpenCV is a highly optimized library, we opted for a thresholding focused approach:

#Convert our image into Greyscale
#Perform thresholding to build a mask for the foreground and background
#Determine the foreground and background based on the mask
#Reconstruct original image by combining foreground and background

def thresholding(img):
    img =cv2.imread(img)

    # First Convert to Grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(img_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(img,img, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    # cv2.imwrite("foreground.png",foreground)
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("background.png",background)

    # Combine the background and foreground to obtain our final image
    sharpened = background+foreground
    # cv2.imwrite("method_2.png",sharpened)
    return sharpened

def remove_bg(img, file_name=None,output_dir=None):
        # join output filename
        if file_name:
            output_path = os.path.join(output_dir, file_name)

        # convert image to 4-channel
        # img.save('original.png')
        img = img.convert("RGBA")
        # img.save('test2.png')
        newData = []
        for item in img.getdata():
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        if file_name:
            img.save(output_path, "PNG")
        return img

def merge_bg(front,back,location):
    front = front.convert("RGBA")
    back = back.convert("RGBA")
    # Paste the frontImage at (width, height)
    back.paste(front, (location[0], location[1]), front)
    return back