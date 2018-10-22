import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import time 


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 960, 720)

dimx = 640
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


tmp = cv2.imread("target_qrcode/barcode2.png")

target_barcode = cv2.resize(tmp, (96,96))

orb = cv2.ORB_create(nfeatures=100, nlevels = 8, scaleFactor=1.2, patchSize=31, edgeThreshold=15, fastThreshold = 1)

kp_target = orb.detect(target_barcode, None)

print([kp.size for kp in kp_target])

#kp_target = cv2.KeyPoint(64,64, 63.0)
kp_target, des_target = orb.compute(target_barcode,kp_target)




print(len(kp_target))



for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    
    t0 = time()
    
    orb = cv2.ORB_create(nfeatures=800, nlevels = 8, scaleFactor=2.0, patchSize=31, edgeThreshold=15, fastThreshold = 5)
    
    kp = orb.detect(image, None)
    
    t1 = time()
    
    kp, des = orb.compute(image,kp)
    
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_5 = bf.knnMatch(des, des_target, k=5)

    matches = reduce((lambda x,y : x+y), matches_5)
    
    
    print(kp[0])

    matches = sorted(matches,key= lambda x:x.distance)
    
    match_good = []
    
    for ind in xrange(len(matches)):
        if matches[ind].distance > 65:
            match_good = matches[:ind]
            break
    
    
    #print([x.distance for x in matches])
    
    good_kp = [kp[x.queryIdx] for x in match_good]

    t2 = time()
    #img2 = cv2.drawKeypoints(image,good_kp,None,color=(0,255,0), flags=0)
    #img2 = cv2.drawKeypoints(target_barcode,kp_target,None,color=(0,255,0), flags=0)
    img3 = cv2.drawMatchesKnn(image, kp, target_barcode, kp_target, [match_good], None, flags=2)
    cv2.imshow("image", img3)
    
    t3 = time()
    
    rawCapture.truncate(0)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    print("FPS", 1.0/(t3-t0),  t3-t0, t1-t0, t2-t1, t3-t2)
    
    c=c+1
    
    
    