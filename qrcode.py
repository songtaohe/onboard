import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import time 


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 960, 720)

dimx = 480
dimy = 360

camera = PiCamera(sensor_mode=5)

camera.vflip = True
camera.hflip = True
camera.resolution = (dimx, dimy)


camera.framerate = 30
camera.shutter_speed = 8000
#camera.contrast = 100

#camera.image_effect = ''

rawCapture = PiRGBArray(camera, size=(dimx, dimy))
c = 0


for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    
    t0 = time()
    
    orb = cv2.ORB_create(nfeatures=800, nlevels = 8, scaleFactor=2, patchSize=15, edgeThreshold=15, fastThreshold = 10)
    
    kp = orb.detect(image, None)
    
    t1 = time()
    
    kp, des = orb.compute(image,kp)
    
    if c>0 :
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, des_last)
         
        matches = sorted(matches,key= lambda x:x.distance)
        
        
        good_kp = [kp[x.queryIdx] for x in matches[:min(400,len(matches))] ]
        good_kp2 = [kp_last[x.trainIdx] for x in matches[:min(400,len(matches))] ]
    else:
        good_kp = kp
        good_kp2= kp
        
        
    
    kp_last, des_last = kp, des
    
    
    
    print(kp[0])
    
    
    t2 = time()
    img2 = cv2.drawKeypoints(image,good_kp,None,color=(0,255,0), flags=0)
    img3 = cv2.drawKeypoints(img2,good_kp2,None,color=(255,0,0), flags=0)
    
    cv2.imshow("image", img3)
    
    t3 = time()
    
    rawCapture.truncate(0)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    print("FPS", 1.0/(t3-t0),  t3-t0, t1-t0, t2-t1, t3-t2)
    
    c=c+1
    
    
    