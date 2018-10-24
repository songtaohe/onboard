import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from time import time 
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import sys


output_folder = None

if len(sys.argv)>1 :
    output_folder = sys.argv[1]
else:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280, 700)





dimx = 720*2
dimy = 480*2

# camera = PiCamera(sensor_mode=5)

# camera.vflip = True
# camera.hflip = True
# camera.resolution = (dimx, dimy)


# camera.framerate = 30
# camera.exposure_compensation = 24
# camera.exposure_mode = 'antishake'

# camera.shutter_speed = 20000
#camera.contrast = 100

#camera.image_effect = ''

#rawCapture = PiRGBArray(camera, size=(dimx, dimy))
c = 0

def remove_similar_feature(kp, des, threshold = 40, both=True):
    flag = False
    
    print("number of kp:", len(kp))
    
    mark = [0] * len(kp)
    
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    bf = cv2.BFMatcher()
    #print(des)
    matches_5 = bf.knnMatch(des, des, k=2)
    
    for matches in matches_5:
        if matches[1].distance < threshold:
            if both == True:
                mark[matches[1].queryIdx] = 1
                mark[matches[1].trainIdx] = 1
            else:
                mark[min(matches[1].queryIdx,matches[1].trainIdx)] = 1 

            
    new_kp = []
    new_des = []
    
    for ind in xrange(len(mark)):
        if mark[ind] == 0 :
            new_kp.append(kp[ind])
            new_des.append(des[ind])
        else:
            flag = True
    
    new_des = np.vstack(new_des)
            
    if flag == True and len(new_kp)>0 :
        return remove_similar_feature(new_kp, new_des, threshold)
    else:
        return new_kp, new_des 



def generate_target(filename="target_qrcode/barcode2a.png", size=96):

    tmp = cv2.imread(filename)

    target_barcode = cv2.resize(tmp, (size,size))

    #orb = cv2.ORB_create(nfeatures=200, nlevels = 8, scaleFactor=1.1, patchSize=31, edgeThreshold=15, fastThreshold = 10)
    orb = cv2.xfeatures2d.SIFT_create(nfeatures=150)


    kp_target = orb.detect(target_barcode, None)

    print([kp.size for kp in kp_target])

    #kp_target = cv2.KeyPoint(64,64, 63.0)
    kp_target_, des_target_ = orb.compute(target_barcode,kp_target)

    
                        
    kp_target, des_target = remove_similar_feature(kp_target_, des_target_, 0)


    return target_barcode, kp_target, des_target





kp_target = []
des_target = []


target_barcode, kp1, des1 = generate_target("target_qrcode/barcode2.png", size=96)
_, kp2, des2 = generate_target("target_qrcode/barcode2_blur.png",size=96)
#_, kp3, des3 = generate_target("target_qrcode/barcode2c.png")
#_, kp4, des4 = generate_target("target_qrcode/barcode2e.png")


kp_target = kp1+kp2#+kp3+kp4
des_target = np.vstack([des1, des2])



# target_barcode, kp1, des1 = generate_target("target_qrcode/barcode2d.png", size=128)
# _, kp2, des2 = generate_target("target_qrcode/barcode2b.png",size=128)
# _, kp3, des3 = generate_target("target_qrcode/barcode2c.png",size=128)
# _, kp4, des4 = generate_target("target_qrcode/barcode2e.png",size=128)


# kp_target_ = kp1+kp2+kp3+kp4
# des_target_ = np.vstack([des1, des2, des3, des4])

# kp_target, des_target = remove_similar_feature(kp_target_, des_target_, 80, both=False)





cap = cv2.VideoCapture('outdoor_example3.mov')
#cap = cv2.VideoCapture(0)


#for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
while(cap.isOpened()):
    #image = frame.array

    ret, image_orig = cap.read()


    image_gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

    #ret, image = cv2.threshold(image_gray,224,255,cv2.THRESH_BINARY)
    #image = cv2.adaptiveThreshold(image_gray, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127,2)

    image = image_orig





    key = cv2.waitKey(100) & 0xFF
    
    #print("!!!!!!!!!!!! ", key, c)

    if key == ord("q"):
        break


    if key == ord("c"):
        c = c + 1
        continue

    print(image.shape)

    
    t0 = time()
    
    #orb = cv2.ORB_create(nfeatures=800, nlevels = 6, scaleFactor=1.2, patchSize=31, edgeThreshold=15, fastThreshold = 10)
    orb = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
    kp = orb.detect(image, None)
    

    print(len(kp))

    t1 = time()
    
    kp, des = orb.compute(image,kp)
    
    
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    bf = cv2.BFMatcher()
    matches_5 = bf.knnMatch(des, des_target, k=2)

    if len(matches_5) == 0:

        #rawCapture.truncate(0)
        continue


    matches = reduce((lambda x,y : x+y), matches_5)
    
    
    print(kp[0])

    matches = sorted(matches,key= lambda x:x.distance)
    
    match_good = []
    
    for ind in xrange(len(matches)):
        if matches[ind].distance > 165 or ind > 50:
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
    
    if len(match_good)>3 :
    
        src_pts = np.float32([kp[x.queryIdx].pt for x in match_good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_target[x.trainIdx].pt for x in match_good]).reshape(-1,1,2)
    
        
        #print(dst_pts)
    
    
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 8)
    
        if mask is not None:
            matchesMask = mask.ravel().tolist()
            
            if len(matchesMask) > 2:
        
                h,w = target_barcode.shape[0:2]
                print(h,w)
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

                try:
                    dst = cv2.perspectiveTransform(pts, M)
                
                    print(M)
                    print(dst)
                except:
                    dst = None 
    
   
        
    #print(dst)
    
    #print([x.distance for x in matches])
    
    good_kp = [kp[x.queryIdx] for x in match_good]

    t2 = time()
    #image = cv2.drawKeypoints(image,kp,None,color=(0,255,0), flags=0)
    

    # for k in good_kp :
    #     image = cv2.circle(image, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0,255,0,25),-1)    


    img1 = cv2.drawKeypoints(image_orig,good_kp,None,color=(0,255,0), flags=4)
    #img1 = cv2.drawKeypoints(image,kp,None,color=(0,255,0), flags=2)
    
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
    
    else:
        cv2.imshow("image", img3)
    
    t3 = time()
    
    #rawCapture.truncate(0)
    
    
    
    print("FPS", 1.0/(t3-t0),  t3-t0, t1-t0, t2-t1, t3-t2)
    
    c=c+1
    
    
    
