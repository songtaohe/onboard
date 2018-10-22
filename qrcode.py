import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import time 
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import sys


output_folder = None

if len(sys.argv)>1 :
    output_folder = sys.argv[1]

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 960, 720)

dimx = 720*2
dimy = 480*2

camera = PiCamera(sensor_mode=5)

camera.vflip = True
camera.hflip = True
camera.resolution = (dimx, dimy)


camera.framerate = 30
#camera.shutter_speed = 8000
#camera.contrast = 100

#camera.image_effect = ''

rawCapture = PiRGBArray(camera, size=(dimx, dimy))
c = 0


tmp = cv2.imread("target_qrcode/barcode2.png")

target_barcode = cv2.resize(tmp, (96,96))

orb = cv2.ORB_create(nfeatures=100, nlevels = 6, scaleFactor=1.2, patchSize=31, edgeThreshold=15, fastThreshold = 1)

kp_target = orb.detect(target_barcode, None)

print([kp.size for kp in kp_target])

#kp_target = cv2.KeyPoint(64,64, 63.0)
kp_target, des_target = orb.compute(target_barcode,kp_target)




print(len(kp_target))



for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
    image = frame.array
    
    t0 = time()
    
    orb = cv2.ORB_create(nfeatures=800, nlevels = 6, scaleFactor=2.0, patchSize=31, edgeThreshold=15, fastThreshold = 20)
    
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
    

    #X = np.float32([kp[x.queryIdx].pt for x in match_good]).reshape(-1,2)
    #X = np.float32([x.pt for x in kp]).reshape(-1,2)

    #print(X)

    #bandwidth = estimate_bandwidth(X, quantile=0.1)
    
    
    #print(bandwidth)
    
    
    #ms = MeanShift(bandwidth = 128, bin_seeding=True, cluster_all = False)
    #ms.fit(X)
    
    #labels = ms.labels_
    #cluster_centers = ms.cluster_centers_
    
    #print(labels)
    #print(cluster_centers)
    
    dst = None
    
    if len(match_good)>5 :
    
        src_pts = np.float32([kp[x.queryIdx].pt for x in match_good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_target[x.trainIdx].pt for x in match_good]).reshape(-1,1,2)*10
    
        
        #print(dst_pts)
    
    
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 2.5)
    
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            
            if len(matchesMask) > 5:
        
                h,w = target_barcode.shape[0:2]
                print(h,w)
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts*10, M)
    
                print(M)
                print(dst)
    
   
        
    #print(dst)
    
    #print([x.distance for x in matches])
    
    good_kp = [kp[x.queryIdx] for x in match_good]

    t2 = time()
    #image = cv2.drawKeypoints(image,kp,None,color=(0,255,0), flags=0)
    
    img1 = cv2.drawKeypoints(image,good_kp,None,color=(0,255,0), flags=2)
    
    #for cls in  cluster_centers:
    #    img1 = cv2.circle(img1, (int(cls[0]), int(cls[1])), 32, 255,)    
    
    if dst is not None:
        img2 = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        img2 = img1
    #img2 = cv2.drawKeypoints(target_barcode,kp_target,None,color=(0,255,0), flags=0)
    img3 = cv2.drawMatchesKnn(img2, kp, target_barcode, kp_target, [match_good], None, flags=2)
    
    #img3 = cv2.drawMatchesKnn(img2, kp, target_barcode, kp_target, [match_good], None, flags=2)
    
    if output_folder is not None :
        cv2.imwrite(output_folder+"/img%05d.jpg" % c, img3)
    
    
    cv2.imshow("image", img3)
    
    t3 = time()
    
    rawCapture.truncate(0)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
    print("FPS", 1.0/(t3-t0),  t3-t0, t1-t0, t2-t1, t3-t2)
    
    c=c+1
    
    
    