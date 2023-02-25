import math
import numpy as np
import random

from scipy.spatial.distance import cdist
from utils import pad, unpad

'''
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 !!! NE MODIFIEZ PAS LE CODE EN DEHORS DES BLOCS TODO. !!!
 !!!  L'EVALUATEUR AUTOMATIQUE SERA TRES MECHANT AVEC  !!!
 !!!            VOUS SI VOUS LE FAITES !               !!!
 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

def fit_transform_matrix(p0, p1):
    """ Calcul la matrice de transformation H tel que p0 * H.T = p1

   import  Indication:
        Vous pouvez utiliser la fonction "np.linalg.lstsq" ou
        la fonction "np.linalg.svd" pour résoudre le problème.

    Entrées :
        p0 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points à transformer
        p1 : un tableau numpy de dimension (M, 2) contenant
             les coordonnées des points destination

    Sortie :
        H : la matrice de transformation de dimension (3, 3)
    """

    assert (p1.shape[0] == p0.shape[0]),\
        'Nombre différent de points en p1 et p2'

    H = None
    
    #TODO 1 : Calculez la matrice de transformation H. Notez que p0 et p1
    #         sont des tableaux de coordonnées organisés en lignes.
    #          c-à-d.  p0[i,:] = [p0line_i, p0col_i]
    #             et   p1[j,:] = [p1line_j, p1col_i]
    # TODO-BLOC-DEBUT    
    # raise NotImplementedError("TODO 1 : dans panorama.py non implémenté")
    num_matches = p0.shape[0]
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    #w = np.zeros(num_rows)
        
    for i in range(num_matches):
        (x_i, y_i) = tuple(p0[i])
        (p_i, q_i) = tuple(p1[i])

        A[2*i,   :] = [x_i , y_i , 1 ,   0 ,  0 , 0 , -p_i*x_i , -p_i*y_i , -p_i]
        A[2*i+1, :] = [  0 ,   0 , 0 , x_i , y_i, 1 , -q_i*x_i , -q_i*y_i , -q_i]
        
        #w[2*i  ] = p_i
        #w[2*i+1] = q_i        
    
    U, s, Vt = np.linalg.svd(A)
    
    #H = np.linalg.lstsq(A[:,:-1], w , rcond=None)[0]
    #H = pad(H.reshape(1,-1)).reshape(3,3).T
    
    H  = Vt[8,:].reshape(3,3).T    
    # TODO-BLOC-FIN

    return H

def ransac(keypoints1, keypoints2, matches, n_iters=500, threshold=0.5):
    """
    Utilisez RANSAC pour trouver une transformation projective robuste

        1. Sélectionnez un ensemble aléatoire de correspondances
        2. Calculez la matrice de transformation
        3. Calculer les bonnes correspondances (inliers)
        4. Gardez le plus grand ensemble de bonnes correspondances
        5. En final, recalculez la matrice de transformation sur tout l'ensemble
           des bonnes correspondances

    Entrées :
        keypoints1 -- matrice M1 x 2, chanque rangée contient les coordonnées d'un point-clé dans image1
        keypoints2 -- matrice M2 x 2, chanque rangée contient les coordonnées d'un point-clé dans image2
        matches -- matrice N x 2, chaque rangée représente une correspondance
            [indice dans keypoint1, indice dans keypoint 2]
        n_iters -- le nombre d'itérations dans RANSAC
        threshold -- le seuil pour trouver des bonnes correspondances

    Sorties :
        H -- une estimation robuste de la transformation des points keypoints1 en points keypoints2
        matches[max_inliers] -- les bonnes correspondances
    """
    # indices des bonnes correespondances dans le tableau 'matches' 
    max_inliers = []
    
    # matrice to transformation Homographique
    H = None
    
    # initialisation du générateur de nombres aléatoires
    # fixé le seed pour pouvoir comparer le résultat retourné par 
    # cette fonction par rapport à la solution référence
    #rand = np.random.default_rng(seed=131)
    random.seed(131)
    
    #TODO 2 : Implémentez ici la méthode RANSAC pour trouver une transformation robuste
    # entre deux images image1 et image2.
    # TODO-BLOC-DEBUT    
    # raise NotImplementedError("TODO 2 : dans panorama.py non implémenté")
    
    num_matches = len(matches)
    #all_idxs = np.arange( num_matches )

    sequence  = list( range(num_matches) )
        
    for k in range(n_iters):         
        #np.random.shuffle(all_idxs)                        
        
        maybe_idxs = random.sample(sequence, k=4)        
        test_idxs  = [elem for elem in sequence if elem not in maybe_idxs]
                  
        maybe_idxs = np.array(maybe_idxs)
        test_idxs  = np.array(test_idxs)
        
        #maybe_idxs = all_idxs[:4]
        #test_idxs  = all_idxs[4:]                
            
        matched1 = keypoints1[ matches[maybe_idxs,0] ]
        matched2 = keypoints2[ matches[maybe_idxs,1] ]

        # initial guess of H
        H = fit_transform_matrix(matched1, matched2) 

        # Homography matrix H should not be singular!
        # this is due to bad choice of guess keypoints...  try again !
        if math.isclose(np.linalg.det(H), 0.0):
            continue
            
        # set initial guess of matched key-points as inliers
        inlier_indices = maybe_idxs.tolist()

        # transform to homogeneous coordinates
        pk1 = pad(keypoints1[ matches[test_idxs,0] ])
        pk2 = pad(keypoints2[ matches[test_idxs,1] ])

        # apply  H^T transform to pk1 keypoints 
        tk1 = pk1.dot(H)         
        
        # divide transformed tk1 points by their z-component if not zero                
        ind = np.where( np.logical_not( np.isclose(tk1[:,2], 0.0 ) ) )[0]
        tk1[ind,:] /= tk1[ind,2].reshape(-1,1)
        
        # compute euclidian  distance between transformed 
        # tk1 and reference pk2 points
        vals = np.linalg.norm(tk1 - pk2, axis=1)
          
        # get indices of matching points
        ind  = np.where(vals < threshold)[0]
        
        # add to inliers list
        inlier_indices.extend( test_idxs[ind].tolist() )
                
        # keeo best inliers list
        if  len(inlier_indices) > len(max_inliers):
            max_inliers = inlier_indices

    # retrieve the list of inlier matches
    matched1 = keypoints1[matches[max_inliers,0]]
    matched2 = keypoints2[matches[max_inliers,1]]

    # recompute the transform matrix using inliers only
    H = fit_transform_matrix(matched1, matched2)
    
    # TODO-BLOC-FIN
    
    return H, matches[max_inliers]


def get_output_space(imgs, transforms):
    """
    Ceci est une fonction auxilière qui prend en entrée une liste d'images et
    des transformations associées et calcule en sortie le cadre englobant
    les images transformées.

    Entrées :
        imgs -- liste des images à transformer
        transforms -- liste des matrices de transformation.

    Sorties :
        output_shape (tuple) -- cadre englobant les images transformées.
        offset -- un tableau numpy contenant les coordonnées du coin minimal du cadre
    """

    assert (len(imgs) == len(transforms)),\
        'Different number of images and associated transforms'

    output_shape = None
    offset = None

    all_corners = []

    for img, H in zip(imgs, transforms):
        r, c, _ = img.shape        
        corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])

        warped_corners = pad(corners.astype(np.float)).dot(H).T
        all_corners.append( unpad( np.divide(warped_corners, warped_corners[2,:] ).T ) )
                          
    # Trouver l'étendue des images déformée
    all_corners = np.vstack(all_corners)

    # La forme globale du cadre sera max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = corner_max - corner_min

    # Conversion en nombres entiers avec np.ceil et dtype
    output_shape = tuple( np.ceil(output_shape).astype(np.int) )
    
    # Trouver le deplacement du coin inférieur du cadre par 
    # rapport à l'origine (0,0)
    offset = corner_min

    return output_shape, offset


def warp_image(img, H, output_shape, offset, method=None):
    """
    Deforme l'image img grace à la transformation H. L'image déformée
    est copiée dans une image cible de dimensions 'output_shape'.

    Cette fonction calcule également les coefficients alpha de l'image
    déformée pour un fusionnement ultérieur avec d'autres images.

    Entrée :
        img -- l'image à déformer
        H -- matrice de transformation
        output_shape -- dimensions de l'image transformée
        offset --  position du cadre de l'image tranformée.
        method -- paramètre de sélection de la méthode de calcul des
                  coéfficients alpha.
                  'hlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal à partir du centre jusqu'au
                              bord de l'image
                  'vlinear' -- le alpha varie linéairement de 1.0 à 0.0
                              en vertical à partir du centre jusqu'au
                              bord de l'image
                  'linear' -- le alpha varie linéairement de 1.0 à 0.0
                              en horizontal et en vertical à partir du
                              centre jusqu'au bord de l'image
                   None -- le alpha des pixels est égale à 1.0

    Sortie :
        img_warped (np.float32) -- l'image déformée de dimensions output_shape.
                                   Les valeurs des pixels doivent être dans la
                                   plage [0..1] pour pouvoir visualiser les
                                   résultats avec plt.show(...)

        mask -- tableau numpy de booléens indiquant les pixels valides
                dans l'image de sortie "img_warped"
    """

    image_warped = None
    mask = None
    
    #TODO 3 et 4 : Dans un premier temps (TODO 3), implémentez ici la méthode 
    # qui déforme une image img en applicant dessus la matrice de transformation H. 
    # Vous devez utiliser la projection inverse pour votre implémentation.
    # Pour cela, commencez d'abord par translater les coordonnées de l'image 
    # destination  avec "offset" avant d'appliquer la transformation
    # inverse pour retrouver vos coordonnées dans l'image source.

    # TODO 4 : Dans un deuxième temps, implémentez la partie du code dans cette
    # fonction (controlé avec le paramètre method donné ci-dessus) qui calcule 
    # les coefficients du canal alpha de l'image transformée.
    # TODO-BLOC-DEBUT    
    # raise NotImplementedError("TODO 3,4 : dans panorama.py non implémenté")
    
    ih, iw, c = img.shape

    ( h , w ) = output_shape

    x = np.arange(0, w, 1) + offset[1]
    y = np.arange(0, h, 1) + offset[0]

    xx , yy = np.meshgrid(x, y)
    warped  = pad(np.vstack([yy.flatten(), xx.flatten()]).T).dot(np.linalg.inv(H)).T
    warped  = unpad((np.divide( warped, warped[2,:])).T).T
    yy     = warped[0,:].reshape(h,w)
    xx     = warped[1,:].reshape(h,w)

    top    = np.floor(yy).astype('int')
    bottom = np.floor(yy+1).astype('int')
    left   = np.floor(xx).astype('int')
    right  = np.floor(xx+1).astype('int')
    """
               tl <- u1 -> xx <--- u2 ---> tr
               ^            |              ^
               |            |              |
               v1           |              v1
               |            |              |
               v            |              v
               yy ----------o---------------
               ^            |              ^
               |            |              |
               |            |              |
               v2           |              v2
               |            |              |
               |            |              |
               v            |              v
               bl <- u1 ->  | <--- u2 ---> br
    """
    u1 = xx - left
    u2 = right - xx

    v1 = yy - top
    v2 = bottom - yy

    u1v1 = u1 * v1  # weight for br
    u1v2 = u1 * v2  # weight for tr
    u2v1 = u2 * v1  # weight for bl
    u2v2 = u2 * v2  # weight for tl

    mask = (xx < 0) + (xx > (iw-1)) + (yy < 0) + (yy > (ih-1))
    u1v1[mask]=0.
    u1v2[mask]=0.
    u2v1[mask]=0.
    u2v2[mask]=0.

    left[mask]=0
    right[mask]=0
    top[mask]=0
    bottom[mask]=0
    mask = ~mask

    ximg = np.zeros((ih+1, iw+1, c)) # expand image 
    ximg[:ih, :iw, :] = img / 255.

    img_warped   = u2v2[:, :, None] * ximg[top, left,:]
    img_warped  += u2v1[:, :, None] * ximg[bottom, left,:]
    img_warped  += u1v2[:, :, None] * ximg[top, right,:]
    img_warped  += u1v1[:, :, None] * ximg[bottom, right,:]

    ## now we compute the alphas
    horz = np.ones((ih,iw))
    if method == 'hlinear' or method == 'linear':
        mid = iw // 2
        first_half  = np.linspace(0., 1., mid + 2)
        second_half = np.linspace(1., 0., iw - mid + 1)

        hat   = np.concatenate([first_half[1:],second_half[1:-1]])
        horz  = np.tile(hat, (ih, 1))

    vert = np.ones((ih,iw))
    if method == 'vlinear' or method == 'linear':
        mid = ih // 2
        first_half  = np.linspace(0., 1., mid + 2)
        second_half = np.linspace(1., 0., ih - mid + 1)

        hat   = np.concatenate([first_half[1:],second_half[1:-1]])
        vert  = np.tile(hat, (iw, 1)).T

    weights          = np.zeros((ih+1,iw+1))
    weights[:ih,:iw] = horz * vert

    alpha   = u2v2 * weights[top, left]
    alpha  += u2v1 * weights[bottom, left]
    alpha  += u1v2 * weights[top, right]
    alpha  += u1v1 * weights[bottom, right]

    img_warped = np.dstack((img_warped, alpha))
    # TODO-BLOC-FIN
    
    return img_warped, mask


def stitch_multiple_images(imgs, keypoints_list, matches_list, imgref=0, blend=None):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        keypoints_list: List of detected keypoints for each image in imgs
        matches_list: List of keypoints matches between image i and i+1        
        imgref: index of reference image in the list
        blend: blending method to use to make the panorama, valid arguments should be
               None
               'vlinear'
               'hlinear'
               'linear'

    Returns:
        panorama: Final panorma image in coordinate frame of reference image 
    """
    panorama = None
    
    #TODO BONUS : Votre implémenation ici
    # TODO-BLOC-DEBUT    
    # raise NotImplementedError("TODO BONUS : dans panorama.py non implémenté")
    transforms = compute_Transforms(keypoints_list, matches_list, imgref)
    output_shape, offset = get_output_space(imgs, transforms)
    
    panorama = np.zeros((output_shape[0], output_shape[1], 3))
    weights= np.zeros(output_shape)
    mask = np.zeros(output_shape).astype('bool')
        
    for img, H in zip(imgs,transforms):        
        warped, m = warp_image(img, H, output_shape, offset, blend)
        alpha = warped[:,:,3]
        panorama += alpha[:,:,np.newaxis] * warped[:,:,:3]
        weights += alpha
        mask = mask + m
 
    # Normalize through division by `overlap` - but ensure the minimum is 1
    panorama[mask,:]= panorama[mask,:] / weights[mask,np.newaxis] 
    # TODO-BLOC-FIN

    return panorama


def compute_Transforms(keypoints_list, matches_list, imgref=0):
    """
      0 --> 1 -->  2 -->  ... -->  i-1 -->  i  <-- i+1  <--  ...  <--  N-1
    """

    assert ( imgref >=0 and imgref < len(keypoints_list)),\
        'imgref out of range'

    transforms = []

    t= np.eye(3)
    
    # i goes from  ( imgref down to 1 ) inclusive    
    for i in range(imgref,0,-1):
        transforms.insert(0, t)        

        #project img[i-1] on img[i]
        H , _ = ransac(keypoints_list[i-1], keypoints_list[i], matches_list[i-1])

        t = t.dot( H )
        #t = H.dot(t)
        t = t /np.linalg.norm(t)

    #project img[0] on img[imgref]
    transforms.insert(0, t)

    t = np.eye(3)

    for i in range(imgref,len(keypoints_list)-1):
        #project img[i+1] on img[i]
        #H , _ = ransac(keypoints_list[i], keypoints_list[i+1], matches_list[i])

        #project img[i+1] on img[imgref]
        #t = t.dot( np.linalg.inv(H) )

        #inverse Homography
        H , _ = ransac(keypoints_list[i+1], keypoints_list[i], matches_list[i][:,[1,0]])
        
        #project img[i+1] on img[imgref]
        #t = t.dot( np.linalg.inv(H) )
        
        t = t.dot( H )        
        #t = H.dot(t)
        t = t /np.linalg.norm(t)
                
        transforms.append(t)

    return transforms

