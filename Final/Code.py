'''                             CAP5415 - Fall 2015
                                  Final Project
                            Object Detection for Vehicles
                                   Syed Ahmed
  						     .'  '.			   
   						    | STOP |	     //^\\\   ,;;;,                 
  					    	     '.__.'         ((-_-))) (-_- ;                 
    						       ||   	     )))(((   >..'.    .:.     .--. 
    						       ||   	    ((_._ )  /.   .|  :-_-;   /-_-))
			\`.                            ||   	    _))C ((_//| R ||  ,`-'.   ))-((
            _...--.      |_'.__                        ||     	    `(    )`' |___|),;, C \\_/,`V ))
        .-`___ _  '----'    __```''--.,_               ||     	      \  /    | | |`' |___(/-'|___()
        .' .' _ '.`\    _.-"``  `-._.-._.-;            ||              )(     | | |   | | |   | | | 
    _/   | (_) |  '.-'          | (_) |__`\            ||             /__\    |_|_|   |_|_|   |_|_| 
    '""""'.___.'""""""""""""""""'.___.'---'        ^^^^^^^^^^^        `''     `-'-'   `-'-'   `-'-' 
'''
# Dataset for traffic signs used from The Laboratory for Intelligent and Safe Auto-mobiles (LISA), CVRR, University of California, San Diego.
# Dataset for Cars used from The Laboratory for Intelligent and Safe Auto-mobiles (LISA), CVRR, University of California, San Diego.
# Dataset for Pedestrian used from CalTech Pedestrian Database, California Institute of Technology, Pasadena.
import numpy as np
import cv2

#========================== Detect Vehicles ===========================Start=======#
def detect_vehicles( # the procedure detects the vehicles with the help of the used Haar classifier
image # The frame of the video or the image to which the algorithm is to be applied
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the input RGB image to gray....
    cascade = cv2.CascadeClassifier('haar/vehicleDetect.xml') # extracting the cascade features using Haar classifiers...
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=6,minSize=(35,35)) # This is an inbuilt cv function that is 
    #used to detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles inside the object 'features'.
    '''The object 'features' here gives the left top corner pixel of the detected cars on the image in the following format
    [x index,y index, length occupied from the index x, length occupied from y]'''
    # note that we have the location of only one pixel (the one on the left top corner of the object) here 
    # and to draw the box that highlights our object, we need the location of another pixel in diagonally opposite direction
    # (the one on the right bottom corner of the object)....
    
    if len(features) == 0: # if no features are found, we simply return an empty array.
        return []
    else: # when the required features are detected, we execute the following steps....
        features[:, 2:] += features[:, :2] # using the horizontal and vertical lengths occupied by the object, we calculate the location of
        # of the pixel at the right bottom of the object....
        # the object 'features' now has data in the following format 
        # [x LTC, y LTC, x RBC, y RBC] || LTC--> Left top corner || RBC --> Right bottom corner
        '''since we now have the accurate position of the object on our image, we now have to mark a rectangle around the object'''
        detection_box = [] # this array stores the length of the rectangle to be plotted with respect to the LTC pixel.
        for i in features: # retaining the previous state of features in detection_box
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))# in order to have the length and breadth of the rectangle        
        # Now we have the length and breadth of the rectangle, we use the cv inbuilt function 'rectangle' to highlight the detected object
        for j in detection_box:
                x1, y1, w1, h1 = j #format --> [x index of LTC pixel, y index, width of the rectangle, height of the rectangle]
                cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0,255,0), 2)
                cv2.putText(frame, "Vehicle", (x1, y1+h1+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (0,255,0))
#========================== Detect Vehicles ===========================End=======#

#========================== Detect pedestrians ===========================Start=======#
def detect_pedestrians( # the procedure detects Pedestrians with the help of the used Haar classifier
image # The frame of the video or the image to which the algorithm is to be applied
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the input RGB image to gray....
    cascade = cv2.CascadeClassifier('haar/MyPedestrian.xml') # extracting the cascade features using Haar classifiers...
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=15,minSize = (26,74),maxSize = (70,174)) 
    
    if len(features) == 0: # if no features are found, we simply return an empty array.
        return []
    else: # when the required features are detected, we execute the following steps....
        features[:, 2:] += features[:, :2] 
        detection_box = [] # this array stores the length of the rectangle to be plotted with respect to the LTC pixel.
        for i in features: # retaining the previous state of features in detection_box
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))
        for j in detection_box:
                x1, y1, w1, h1 = j #format --> [x index of LTC pixel, y index, width of the rectangle, height of the rectangle]
                cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0,0,255), 2)
                cv2.putText(frame, "Pedestrian", (x1, y1+h1+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (0,0,255))
#========================== Detect pedestrians ===========================End=======#

#========================== Detect Traffic Signs ===========================Start=======#
def detect_signs( # the procedure detects traffic signs with the help of the used Haar classifier
image # The frame of the video or the image to which the algorithm is to be applied
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the input RGB image to gray....
    '''The Haar classifier used here was trained with traffic signs with rectangular shapes like speed limits, turn restrictions,round about etc.,'''
    cascade = cv2.CascadeClassifier('haar/rectangle.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=1,flags=0, minSize=(35,35),maxSize=(90,90))
    if len(features) == 0: # if no features are found, we simply return an empty array.
        return []
    else: # when the required features are detected, we execute the following steps....
        features[:, 2:] += features[:, :2] # using the horizontal and vertical lengths occupied by the object, we calculate the location of
        detection_box = [] # this array stores the length of the rectangle to be plotted with respect to the LTC pixel.
        for i in features: # retaining the previous state of features in detection_box
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))# in order to have the length and breadth of the rectangle        
        # Now we have the length and breadth of the rectangle, we use the cv inbuilt function 'rectangle' to highlight the detected object
        for j in detection_box:
                x1, y1, w1, h1 = j #format --> [x index of LTC pixel, y index, width of the rectangle, height of the rectangle]
                cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0,255,255), 2)
                cv2.putText(frame, "Sign", (x1, y1+h1+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (0,255,255))

def detect_signs2( # the procedure detects traffic signs with the help of the used Haar classifier
image # The frame of the video or the image to which the algorithm is to be applied
):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Converting the input RGB image to gray....
    '''The classifier used here is was trained with traffic signs of octagonal and rhombus shaped like stop and pedestrian signs
    '''
    cascade = cv2.CascadeClassifier('haar/rhombus.xml')
    features = cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=1,minSize=(36,36),maxSize=(48,48))
        
    if len(features) == 0: # if no features are found, we simply return an empty array.
        return []
    else: # when the required features are detected, we execute the following steps....
        features[:, 2:] += features[:, :2] # using the horizontal and vertical lengths occupied by the object, we calculate the location of
        detection_box = [] # this array stores the length of the rectangle to be plotted with respect to the LTC pixel.
        for i in features: # retaining the previous state of features in detection_box
            detection_box.append((i[0], i[1], i[2] - i[0], i[3] - i[1]))# in order to have the length and breadth of the rectangle        
        # Now we have the length and breadth of the rectangle, we use the cv inbuilt function 'rectangle' to highlight the detected object
        for j in detection_box:
                x1, y1, w1, h1 = j #format --> [x index of LTC pixel, y index, width of the rectangle, height of the rectangle]
                cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0,255,255), 2)
                cv2.putText(frame, "Sign", (x1, y1+h1+13),cv2.FONT_HERSHEY_TRIPLEX, .5, (0,255,255))
#========================== Detect Traffic Signs ===========================End=======#

# using the cv function to capture frames from the input video....
video = cv2.VideoCapture('vid.avi') # location of the input video

while(video.isOpened()): #the following loop runs as long as there are frames to be read....
    ret, frame = video.read() # the array 'frame' represents the current frame from the video and the variable ret is used to check if the 
                                # frame is read. ret gives True if the frame is read else gives false
    if frame is None:
        cv2.destroyAllWindows() # if there are no more frames, close the display window....
        break
    else: # at each frame read....
        detect_vehicles(frame) # detect the vehicles in the current frame
        detect_pedestrians(frame) # detect the pedestrians in the current frame
        detect_signs(frame) # detect the rectangular traffic signs in the current frame
        detect_signs2(frame) # detect the octagonal traffic signs in the current frame
        #uncomment the below lines if you want to check the output frame by frame
        '''
        cv2.waitKey(0)
        if 0xFF in (ord('p'),ord('l')):
            pass
        '''
        cv2.imshow('frame',frame) # display the current frame after running the detection algorithm....
    if cv2.waitKey(1) & 0xFF in (ord('q'),0x1B,0x0D): # if 'Esc','q' or 'Enter' keys are pressed on the keyboard, we exit the loop.
        break
video.release() # When everything done, release the capture...
cv2.destroyAllWindows() # closing the display window automatically...