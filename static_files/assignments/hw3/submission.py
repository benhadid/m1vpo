import math
import numpy as np
import helper as hlp
import numpy.linalg as la
from   scipy import signal

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

def eight_point(pts1, pts2, M):
    """
    TODO2.1
       Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
    """    
    
    assert (pts1.shape[0] == pts2.shape[0]),\
        'Nombre différent de points en pts1 et pts2'

    assert (M > 0.),\
        'Le paramètre M doit étre > 0'
    
    F = None    
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2.1 : dans submission.py non implémenté")    
    # TODO-BLOC-FIN

    return F


def epipolar_correspondences(im1, im2, F, pts1):
    """
    TODO 2.2
       Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
    """
    
    pts2 = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2.2 : dans submission.py non implémenté")
    # TODO-BLOC-FIN
    
    return pts2

    
def essential_matrix(F, K1, K2):
    """
    TODO 2.3
       Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
    """    

    assert (F.shape == (3,3) and K1.shape == (3,3) and K2.shape == (3,3)),\
        'Les matrices F, K1 et K2 doivent être de dimensions : 3 x 3'

    E = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2.3 : dans submission.py non implémenté")    
    # TODO-BLOC-FIN

    return E


    
def triangulate(P1, pts1, P2, pts2):
    """
    TODO 2.4
       Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
    """    
    
    assert (P1.shape == (3,4) and P2.shape == (3,4) ),\
        'Les matrices P1 et P2 doivent être de dimensions : 3 x 4'

    pts3d = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2.4 : dans submission.py non implémenté")    
    # TODO-BLOC-FIN
    
    return pts3d


def ComputeExtrinsic( K1, K2, E, pts1, pts2 ):
    """
    TODO 2.5
    Compute P1 and P2 and the error of reprojection
    This function should call triangulate(...)
    [I] K1, camera matrix 1 (3x3 matrix)
        K2, camera matrix 2 (3x3 matrix)
        E, the essential matrix (3x3 matrix)
        pts1, points in image 1 (Nx2 matrix)
        pts2, points in image 2 (Nx2 matrix)
    [O] R1, camera 1 rotation matrix (3x3 matrix)
        t1, camera 1 translation vector (3x1 matrix)
        R2, camera 2 rotation matrix (3x3 matrix)
        t2, camera 2 translation vector (3x1 matrix)   
        pts3d, 3D points in space (Nx3 matrix)
        err, reprojection error
    """
    R1 = None
    t1 = None
    R2 = None
    t2 = None
    pts3d = None
    err = None
        
    R2t2 = hlp.camera2(E)   # R2t2[i] (avec i = 0..3) sont les quatre matrices extrinsèques possibles

    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2.5 : dans submission.py non implémenté")
    # TODO-BLOC-FIN
    
    return R1, t1, R2, t2, pts3d, err


def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    TODO 3.1
       Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] H1 H2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
    """    

    H1  = None
    H2  = None
    K1p = None
    K2p = None    
    R1p = None
    R2p = None    
    t1p = None
    t2p = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 3.1 : dans submission.py non implémenté")                                
    # TODO-BLOC-FIN

    return H1, H2, K1p, K2p, R1p, R2p, t1p, t2p

    
def get_disparity(im1, im2, max_disp, win_size):
    """
    TODO 3.2
       Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
    """    
    
    dispM = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 3.2 : dans submission.py non implémenté")
    # TODO-BLOC-FIN

    return dispM
    
    
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    TODO 3.3
       Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
    """   
    
    depthM = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 3.3 : dans submission.py non implémenté")    
    # TODO-BLOC-FIN
    
    return depthM


def estimate_pose(x, X):
    """
    TODO 4.1
       Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
    """    
    P = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 4.1 : dans submission.py non implémenté")        
    # TODO-BLOC-FIN

    return P

    
def estimate_params(P):
    """
    TODO 4.2
       Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
    """    
    K = None
    R = None
    t = None
    
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 4.2 : dans submission.py non implémenté")
    # TODO-BLOC-FIN
    
    return K, R, t
