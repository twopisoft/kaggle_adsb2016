import numpy as np
import cv2
import scipy.io
import shutil
import random
import os
import sys
from segment_tut import *
import matplotlib.pyplot as plt

def init_snake(x,y,r):
    snake_init_x = np.array([ x + r * np.cos(th) for th in np.arange(0,2 * np.pi,np.pi/8) ])
    snake_init_y = np.array([ y + r * np.sin(th) for th in np.arange(0,2 * np.pi,np.pi/8) ])
    
    return snake_init_x,snake_init_y

def show_snake(img, snake_x, snake_y, thickness=1, color=255):
    pimg = img.copy()
    s_x = np.append(snake_x, [snake_x[0]]).astype(int)
    s_y = np.append(snake_y, [snake_y[0]]).astype(int)
    for i in range(1,len(s_x)):
        p1 = (s_x[i-1],s_y[i-1])
        p2 = (s_x[i],s_y[i])
        cv2.line(pimg, p1, p2, color, thickness)
        
    plt.imshow(pimg, cmap=plt.cm.gray)
    plt.show()


def thresholding(img, adaptive=True, threshold_value=127, 
                      maxval=255, block_size=7, C=10, 
                      flag=cv2.THRESH_BINARY, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):

    if adaptive:
        return cv2.adaptiveThreshold(img, maxval, adaptiveMethod=method, thresholdType=flag, blockSize=block_size, C=C)
    else:
        return cv2.threshold(img, threshold_value, maxval, flag)[1]

def img_scale(img):
    mx = img.max()
    mn = img.min()
    a = 255.0/(mx - mn)
    b = -255.0/(mx - mn)

    return cv2.convertScaleAbs(img,alpha=a,beta=b)

def snake_calc_all_areas(images, rois, circles):

    def snake_calc_area(time):
        log("Calculating area at time %d..." % time,2)
        areas = {}
    
        for i in range(slices):
            cx,cy = circles[i][0]
            cx,cy=(cx.astype(int),cy.astype(int))
            cr = circles[i][1].astype(int)
            sx,sy = init_snake(cx,cy,0.4*cr)
            img = images[i,time]
            img = cv2.filter2D(img, cv2.CV_32F, kernel)
            img = thresholding(img_scale(img), threshold_value=40, adaptive=False, flag=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            new_sx,new_sy = update_snake(img, sx, sy)
            areas[i] = cv2.contourArea(np.array(zip(new_sx,new_sy)).astype(int))
            
        return areas
    
    kernel = np.array([[1,1,1],[1,-128,1],[1,1,1]]).astype(np.float32)

    (slices, times, _, _) = images.shape
    
    return np.array([snake_calc_area(t) for t in range(times)])

def snake_segment_dataset(dataset):
    images = dataset.images
    dist = dataset.dist
    areaMultiplier = dataset.area_multiplier
    log("Processing dataset %s" % dataset.name,1)
    
    log("Calculating rois...", 2)
    rois, circles = calc_rois(images)
    log("Calculating areas...", 2)
    all_areas = snake_calc_all_areas(images, rois, circles)
    log("Calculating volumes...", 2)
    area_totals = [calc_total_volume(a, areaMultiplier, dist)
                   for a in all_areas]
    log("Calculating ef...", 2)
    edv = max(area_totals)
    esv = min(area_totals)
    ef = (edv - esv) / edv
    log("edv=%f, esv=%f, ef=%f" % (edv,esv,ef),2)
    
    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef

def update_snake_dp(img, init_x, init_y, alpha = 1.0, num_neighbors=3):
    
    def E(curr_node,next_node):
        nx,ny = next_node
        cx,cy = curr_node
        
        ce = gx[cy,cx]**2 + gy[cy,cx]**2
        ne = gx[ny,nx]**2 + gy[ny,nx]**2
        
        e_ext = -abs(ce + ne)
        
        e_int = alpha * ((nx - cx)**2 + (ny - cy)**2)
        
        return e_int + e_ext
    
    sx = init_x.copy()
    sy = init_y.copy()
    
    n = len(sx)
    m = num_neighbors*num_neighbors
    (height,width) = img.shape
    
    x = sx.astype(int)
    y = sy.astype(int)
    offset = m/2
    gx,gy = np.gradient(-img.astype(np.float32))
    
    mask_x = [j for i in range(num_neighbors) for j in range(-(num_neighbors/2),num_neighbors/2+1)]
    mask_y = [j for j in range(-(num_neighbors/2),num_neighbors/2+1) for i in range(num_neighbors)]
    
    energy = np.zeros((n,m))
    pos = np.zeros((n,m), dtype=np.int)
        
    for i in range(0,n-1):

        for j in range(m):
            min_e = np.finfo(np.float32).max
            min_pos = 0

            nxi = x[i+1]
            nyi = y[i+1]
            next_node = (nxi+mask_x[j], nyi+mask_y[j])
            if next_node[0] >= width or next_node[1] >= height:
                next_node = (nxi, nyi)

            for k in range(m):
                cxi = x[i]
                cyi = y[i]

                curr_node = (cxi+mask_x[k], cyi+mask_y[k])
                if curr_node[0] >= width or curr_node[1] >= height:
                    curr_node = (cxi, cyi)

                last_e = 0.0
                if i > 0:
                    last_e = energy[i-1][k]

                eng = last_e + E(curr_node, next_node)
                if (eng < min_e):
                    min_e = eng
                    min_pos = k

            energy[i][j] = min_e
            pos[i][j] = min_pos

    min_final_e = np.finfo(np.float32).max
    min_final_pos = 0
    for j in range(m):
        if (energy[n-2][j] < min_final_e):
            min_final_e = energy[n-2][j]
            min_final_pos = j

    p = min_final_pos
    
    for i in range(n-1,-1,-1):
        sx[i] = sx[i] + mask_x[p]
        sy[i] = sy[i] + mask_y[p]
        if i > 0:
            p = pos[i-1][p]
    
    return sx,sy,min_final_e

def update_snake(img, sx, sy):
    converged = False
    last_e = 0.0
    x = sx.astype(np.float32)
    y = sy.astype(np.float32)
    
    while not converged:
        x,y,e = update_snake_dp(img, x, y, num_neighbors=5, alpha=1.0)
        converged = abs(last_e - e) <= 0.0001
        last_e = e
        
    return x,y

def main():

    random.seed(42)

    settings_f = open("settings.json","r")
    settings = json.load(settings_f)
    settings_f.close()

    auto_segment_all_datasets(segment_fn=snake_segment_dataset, prefix="snake_", ns=settings["ns"],
                              basepath=settings["basepath"], validate=settings["validate"], train=settings["train"])

if __name__ == "__main__":
    main()