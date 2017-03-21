import numpy as np
import cv2


def opticalFlowOverlay(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    output: mask
    """
    feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 5 )
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    image_current_saved = np.copy(image_current)
    image_next_saved = np.copy(image_next)
    
    image_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    image_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(image_current, mask = None, **feature_params)

    p1, st, err = cv2.calcOpticalFlowPyrLK(image_current, image_next, p0, None, **lk_params)


    color = np.random.randint(0, 255, (100, 3))

    mask = np.zeros_like(image_current)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel() # flatten
        c, d = old.ravel()
        mask = cv2.arrowedLine(mask, (a,b), (c, d), color[i].tolist(), 1, 8)
        
        image_next = cv2.circle(image_next_saved, (a, b), 1, color[i].tolist(), -1)
        image_next_fg = cv2.bitwise_and(image_next, image_next, mask = mask)
        
    dst = cv2.add(image_next, image_next_fg)
    return dst

def opticalFlowDenseDim3(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    output: flow_direction + magnitude + original image saturation as (R,G,B) image
    """    
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    hsv = np.zeros((66, 220, 3))

    # set HSV's Saturation value to the original image's saturation value
    hsv_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = hsv_next[:,:,1]
 
    # Flow Parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3 # 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  

    # hue corresponds to direction
    hsv[:,:,0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    hsv = np.asarray(hsv, dtype = np.float32)
    # convert back to RGB
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow


def opticalFlowDenseDim5(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    output: image_difference as data (R,G,B,A,M)
    """
    
    # TODO try gaussian blurring the image first too
    
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    data = np.zeros((66, 220, 5))
    
 
    # Flow Parameters
    flow_mat = cv2.CV_32FC2
    image_scale = 0.4
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.5 # 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # get red differences
    data[:,:,0] = image_current[:,:,0] - image_next[:,:,0]
    
    # get green differences
    data[:,:,1] = image_current[:,:,1] - image_next[:,:,1]
    
    # get blue differences
    data[:,:,2] = image_current[:,:,2] - image_next[:,:,2]
    
    # get hue for data
    data[:,:,3] = ang * (180/ np.pi / 2) * (255/180)
    
    data[:,:,4] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    return data