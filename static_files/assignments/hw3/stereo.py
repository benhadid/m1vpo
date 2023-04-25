import math
import random
import numpy as np
import numpy.linalg as la
import scipy.ndimage as ndi
from scipy import signal

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# transformation coordonnées cartésiennes -> homogènes
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])

# suppression de la dernière coordonnée
unpad = lambda x: x[:,:-1]


def eight_points(pts1, pts2):
    """
    TODO4.1
       Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)           
       [O] F, the fundamental matrix (3x3 matrix)
    """    
    
    assert (pts1.shape[0] == pts2.shape[0]),\
        'Nombre différent de points en pts1 et pts2'
    
    F = None    
    
    # TODO-BLOC-DEBUT    
    pass
    # TODO-BLOC-FIN

    return F


def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=1e-4):
    """
    TODO4.2
       RANSAC pour trouver une transformation projective robuste

       [I] keypoints1,  tableau M1 x 2, chaque rangée contient les coordonnées 
                        d'un point-clé (x_i,y_i) dans image1
           keypoints2,  tableau M2 x 2, chaque rangée contient les coordonnées 
                        d'un point-clé (x'_i,y'_i) dans image2
           matches,     tableau N x 2, chaque rangée représente une correspondance
                        [indice_dans_keypoints1, indice_dans_keypoints2]
           n_iters,     le nombre d'itérations à effectuer pour RANSAC
           threshold,   le seuil pour sélectionner des bonnes correspondances
       [O] F,           une estimation robuste de la matrice Fondamentale F
           goodmatches, tableau max_inliers x 2 contenant les indices des bonnes correspondances 
    """
    
    # Matrice Fondamentale
    F = None
    
    #indices des bonnes correspondances
    goodmatches = None
    
    # Initialisation du générateur de nombres aléatoires
    # fixé le seed pour pouvoir comparer le résultat retourné par 
    # cette fonction par rapport à la solution référence
    random.seed(131)
    
    # TODO-BLOC-DEBUT    
    pass        
    # TODO-BLOC-FIN
                
    return F, goodmatches
        

def epipolar_match(im1, im2, F, pts1, W = 7):
    """
    TODO4.3
       Compute keypoints correspondences using Epipolar geometry
       [I] im1, image 1 (H1xW1x3 matrix)
           im2, image 2 (H2xW2x3 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
    """
    
    assert len(im1.shape) == 3 and len(im2.shape) == 3, '...'
    
    pts2 = None
    
    # TODO-BLOC-DEBUT    
    pass 
    # TODO-BLOC-FIN
    
    return pts2


def estimate_camera_pose( F, principal_point, focal_distance, base_distance ):
    """
    TODO4.4
       Estimate the four possible camera poses
       [I] F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           principal_point, camera's principal point coordinates (1x2 tuple)
           focal_distance, camera's x and y focal lengths (1x2 tuple)
           base_distance, distance betwenn the origins of the cameras (scalar)                      
       [O] K, camera's intrinsic parameters (3x3 matrix)
           Rt_list, camera's extrinsic parameters [R|t] (list of four 4x3 matrices)  
    """
    
    K       = None
    Rt_list = None

    # TODO-BLOC-DEBUT    
    pass
    # TODO-BLOC-FIN

    return K, Rt_list

    
def triangulate(P1, pts1, P2, pts2):
    """
    TODO4.5-1
       Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
    """    
    
    pts3d = None
    
    # TODO-BLOC-DEBUT    
    pass
    # TODO-BLOC-FIN
    
    return pts3d


def check_chirality(K, Rt_list, pts1, pts2):
    """
    TODO4.5-2
       Chirality check
       [I] K, camera intrinsic matrix (3x3 matrix)
           Rt_list, camera's extrinsic parameters [R|t] (list of four 4x3 matrices)             
           pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] Rt, correct camera's extrinsic parameters [R|t] (4x3 matrices)             
           pts3d_list, 3D points in space (list of four Nx3 matrix)
    """    

    Rt = None
    pts3d_list  = None
    
    # TODO-BLOC-DEBUT    
    pass
    # TODO-BLOC-FIN

    return Rt, pts3d_list

def compute_matching_homographies(F, principal_point, pts1, pts2):
    """
    TODO5
       Compute matching homography matrices     
       [I] F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           principal_point, camera's principal point coordinates (1x2 tuple)
           pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] H1, homography transformation matrix for the first image (3x3 matrix)             
           H2, homography transformation matrix for the second image (3x3 matrix)             
    """    

    H1 = None
    H2 = None
    
    # TODO-BLOC-DEBUT    
    pass
    # TODO-BLOC-FIN

    return H1, H2


def compute_disparity(im1, im2, max_disparity, win_size):
    """
    TODO6.1
       Calcul de la carte de disparité
       [I] im1, rectified image 1 (HxWx3 matrix)
           im2, rectified image 2 (HxWx3 matrix)           
           max_disparity, maximum disparity to check (scalar)
           win_size, windows size for block matching (scalar > 0)
       [O] disp, disparity map associated with im1 (HxW matrix)
    """    
    assert im1.shape[0] == im2.shape[0] and \
           im1.shape[1] == im2.shape[1], 'les images doivent avoir des dimensions identiques'
    
    assert 0 < max_disparity and max_disparity < im2.shape[1], 'max_disparity < im1.shape[1]'
    
    disp = None   

    # TODO-BLOC-DEBUT     
    pass
    # TODO-BLOC-FIN
        
    return disp

       
def cross_disparity(im1, im2, disp1, max_disparity, win_size):
    """
    TODO6.2
       Validation de la carte de disparité
       [I] im1, rectified image 1 (HxWx3 matrix)
           im2, rectified image 2 (HxWx3 matrix)           
           disp1, left disparity matrix (HxW matrix)
           max_disparity, maximum disparity to check (scalar)
           win_size, windows size for block matching (scalar > 0)
       [O] disp2, disparity map associated with im2 (HxW matrix)
           dispc, coherent disparity map for im1 (HxW)
    """    
    assert im1.shape[0] == im2.shape[0] and \
           im1.shape[1] == im2.shape[1], 'les images doivent avoir des dimensions identiques'
    
    assert 0 < max_disparity and max_disparity < im2.shape[1], 'max_disparity < im1.shape[1]'
    
    disp2 = None
    dispc = None

    # TODO-BLOC-DEBUT     
    pass
    # TODO-BLOC-FIN
    
    return disp2, dispc


def fill_holes(dispc):    
    """
    TODO6.3
       Disparity holes filling 
       [I] dispc, coherent disparity map with holes (negative values) (HxW)
       [O] dispf, filled disparity map (HxW)
    """    

    dispf = None
        
    # TODO-BLOC-DEBUT     
    pass
    # TODO-BLOC-FIN

    return dispf

