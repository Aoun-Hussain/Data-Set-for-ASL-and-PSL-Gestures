# organize imports
import cv2
import imutils
import numpy as np
import requests

import os



# global variables
bg = None


url = 'http://192.168.0.100:8080/shot.jpg'
#url = 'http://10.163.200.179:8080/shot.jpg'
#url = 'http://10.6.128.154:8080/shot.jpg'

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=15):
    global bg
    # find the absolute difference between background and current frame
    #print("size 2: ", len(image))
    diff = cv2.absdiff(bg.astype("uint8"), image)
    #cv2.imshow("Diff", diff)
    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Thresh", thresholded)

    #print(cv2.RETR_EXTERNAL)
    #print(cv2.CHAIN_APPROX_SIMPLE)

    # get the contours in the thresholded image
    #(_,cnts,_) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, 2)
    cnts = 0
    cnts, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 150, 325, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        gesture_name = input("Please Mention Your Gesture Name: ")
        folder_name = input("Please Mention Your Folder Name: ")
        os.chdir('C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/')
        os.mkdir(folder_name)
        folder_path = 'C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/' + str(folder_name) + '/'
        os.chdir(folder_path)
        # os.mkdir('Thresholded')
        # os.mkdir('Normal')
        os.mkdir(gesture_name)
        # os.mkdir('Channel')

        # thresholded_folder_path = 'C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/'+str(folder_name)+'/'+'Thresholded/'
        # normal_folder_path = 'C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/'+str(folder_name)+'/'+'Normal/'
        #bitwise_folder_path = 'C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/'+str(folder_name)+'/'+str(gesture_name)+'/'
        channel_folder_path = 'C:/Users/Aoun Hussain/Desktop/Vision-Based Sign Language Recognition Equipment for Healthcare A Smart Headset/Code/Dataset/' + str(folder_name) +'/'+str(gesture_name)+'/'

        #print(folder_path)
        i=0
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while(True):
            # get the current frame
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content),dtype = np.uint8)
            img = cv2.imdecode(img_arr, -1)
            frame = img
            #(grabbed, frame) = camera.read()
            
            # resize the framese
            frame = imutils.resize(frame, width=700)

            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)

            # clone the frame
            clone = frame.copy()

##            frame = fgbg.apply(frame)
##            cv2.imshow("Frame", frame)

            # get the height and width of the frame
            (height, width) = frame.shape[:2]

            # get the ROI
            roi = frame[top:bottom, right:left]

            bitwise_image = cv2.bitwise_not(roi)
##            bitwise_channel = cv2.imread(bitwise_image, cv2.IMREAD_UNCHANGED)
##            print('Channel: ', bitwise_channel.shape)
            #cv2.imshow("bitwise", bitwise_image)
            red_channel = bitwise_image[:,:,0]

##            b, g, r = cv2.split(bitwise_image)
##            z = np.zeros_like(g)
##            bitwise_image2 = cv2.merge((z,r,z))
##            cv2.imshow("bitwise2", bitwise_image2)

##            Image = cv2.bitwise_not(frame)
##            cv2.imshow("bitwise", Image)
##            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(bitwise_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
##
##            # to get the background, keep looking till a threshold is reached
##            # so that our running average model gets calibrated
            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
                # segment the hand region
                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand

                    # draw the segmented region and display the frame
                    cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    #cv2.imshow("Thesholded", thresholded)
                   
                    #b, g, r = cv2.split(bitwise_image)
                    #z = np.zeros_like(g)
                    #bitwise_image2 = cv2.merge((z,z,b))
                    #cv2.imshow("bitwise2", bitwise_image2)
                    #print("size", len(bitwise_image))
                    #diff2 = cv2.absdiff(bg.astype("uint8"), bitwise_image)
                    #pix_val = list(bitwise_image.getdata())
                    #print("Pix_value: ", pix_val)
                    #cv2.imshow("diff2", diff2)
                    #thresholded2 = cv2.threshold(diff2, , 255, cv2.THRESH_BINARY)[1]
                    #cv2.imwrite('saved.jpg',thresholded)
            #hand = segment(roi)
            # draw the segmented hand
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

            # increment the number of frames
            num_frames += 1

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            
            # if the user pressed "q", then stop looping
            if keypress == ord("q"):
                break
            if i==25:
                print("start")
            if i>25:
                current_name = str(i-25)
                #img_item = current_name+".jpg"
                #thresholded_path_for_image = thresholded_folder_path+current_name+".jpg"
                #normal_path_for_image = normal_folder_path+current_name+".jpg"
                #bitwise_path_for_image = bitwise_folder_path+current_name+".jpg"
                channel_path_for_image = channel_folder_path+current_name+".jpg"

                #src = cv2.imread('C:/Users/kuldeep.lohana/Desktop/research/Prototype/Gestures/'+str(folder_name)+'/'+'Bitwise/'+current_name+'.jpg', cv2.IMREAD_UNCHANGED)
                #extract red channel
                #print(src.shape)
                #red_channel = src[:,:,0]
                
                
                #print(final_path_for_image)
                #cv2.imwrite(normal_path_for_image, roi)
                #cv2.imwrite(thresholded_path_for_image, thresholded)
                #cv2.imwrite(bitwise_path_for_image, bitwise_image)
                cv2.imwrite(channel_path_for_image, red_channel)
                
                
                
               
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            if i==525:
                break
            else:
                i+=1
# free up memory
camera.release()
cv2.destroyAllWindows()
