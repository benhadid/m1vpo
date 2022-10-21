"""
Devoir 3 - Fonctions auxilières
"""

import cv2 as cv
import numpy as np
import scipy.optimize
import submission as sub
import matplotlib.pyplot as plt


def displayEpipolarF(I1, I2, F):

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F.T @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[1] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()

def epipolarMatchGUI(I1, I2, F):

    sy, sx, sd = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F.T @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[0] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        pc = np.array([[xc, yc]])
        p2 = sub.epipolar_correspondences(I1, I2, F, pc)
        ax2.plot(p2[0,0], p2[0,1], 'ro', MarkerSize=8, linewidth=2)
        plt.draw()


def camera2(E):
    
    # singular values of E must consist of two identical and one zero value
    U,S,Vt = np.linalg.svd(E)  
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(Vt) / m  
    U,S,Vt = np.linalg.svd(E)
    
    #define W
    W = np.array([[0,-1,0], [ 1,0,0], [0,0,1]])
    
    #determinant of R must be positive
    if np.linalg.det(U.dot(W).dot(Vt)) < 0:
        W = -W
            
    M2s = np.zeros([3,4,4])
    #M2s[:,:,0] = np.concatenate([U.dot(W).dot(Vt),    U[:,2].reshape([-1, 1]) / abs(U[:,2]).max() ], axis=1)
    #M2s[:,:,1] = np.concatenate([U.dot(W).dot(Vt),   -U[:,2].reshape([-1, 1]) / abs(U[:,2]).max() ], axis=1)
    #M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(Vt),  U[:,2].reshape([-1, 1]) / abs(U[:,2]).max() ], axis=1)
    #M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(Vt), -U[:,2].reshape([-1, 1]) / abs(U[:,2]).max() ], axis=1)

    M2s[:,:,0] = np.hstack((U @ W   @ Vt, U[:,-1].reshape(-1,1) ))
    M2s[:,:,1] = np.hstack((U @ W   @ Vt,-U[:,-1].reshape(-1,1) ))    
    M2s[:,:,2] = np.hstack((U @ W.T @ Vt, U[:,-1].reshape(-1,1) ))
    M2s[:,:,3] = np.hstack((U @ W.T @ Vt,-U[:,-1].reshape(-1,1) ))
    
    return M2s

        
def _projtrans(H, p):
    n = p.shape[1]
    p3d = np.vstack((p, np.ones((1,n))))
    h2d = H @ p3d
    p2d = h2d[:2,:] / np.vstack((h2d[2,:], h2d[2,:]))
    return p2d


def _mcbbox(s1, s2, H1, H2):
    c1 = np.array([[0,     0, s1[1], s1[1]], 
                   [0, s1[0],     0, s1[0]]])
    
    c1p = _projtrans(H1, c1)
    
    #minx, miny, maxx, maxy
    bb1 = [np.floor(np.amin(c1p[0,:])),  
           np.floor(np.amin(c1p[1,:])),
           np.ceil(np.amax(c1p[0,:])),
           np.ceil(np.amax(c1p[1,:]))]

    # size of the output image 1
    sz1 = [bb1[2] - bb1[0], 
           bb1[3] - bb1[1]]
    
    #
    #
    c2 = np.array([[0,     0, s2[1], s2[1]], 
                   [0, s2[0],     0, s2[0]]])
    
    c2p = _projtrans(H2, c2)

    #minx, miny, maxx, maxy    
    bb2 = [np.floor(np.amin(c2p[0,:])),
           np.floor(np.amin(c2p[1,:])),
           np.ceil(np.amax(c2p[0,:])),
           np.ceil(np.amax(c2p[1,:]))]
    
    # size of the output image 2
    sz2 = [bb2[2] - bb2[0], 
           bb2[3] - bb2[1]]
        
    sz    = np.vstack((sz1, sz2))
    szmax = np.amax(sz, axis=0)
        
    return szmax, bb1[:2], bb2[:2]


def warpStereo(I1, I2, H1, H2):
    sz, tl1, tl2 = _mcbbox(I1.shape, I2.shape, H1, H2)    
    miny = min(tl1[1], tl2[1])     
    T1 = np.array([[1, 0, -tl1[0]], [0, 1, -miny], [0,0,1]])
    T2 = np.array([[1, 0, -tl2[0]], [0, 1, -miny], [0,0,1]])

    sz  = (int(sz[0]), int(sz[1]))
    I1p = cv.warpPerspective(I1, T1 @ H1, sz)
    I2p = cv.warpPerspective(I1, T2 @ H2, sz)
    
    return I1p, I2p, T1, T2
