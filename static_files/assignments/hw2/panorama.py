import math
import numpy as np

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

    Indication:
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
    raise NotImplementedError("TODO 1 : dans panorama.py non implémenté")
    # TODO-BLOC-FIN

    return H


def ransac(keypoints1, keypoints2, matches, n_iters=300, threshold=20):
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

    max_inliers = []
    H = None

    #TODO 2 : Implémentez ici la méthode RANSAC pour trouver une transformation robuste
    # entre deux images image1 et image2.
    # TODO-BLOC-DEBUT    
    raise NotImplementedError("TODO 2 : dans panorama.py non implémenté")
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

    for i in range(len(imgs)):
        r, c, _ = imgs[i].shape
        H = transforms[i]
        corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])

        warped_corners = pad(corners).dot(H).T
        all_corners.append( unpad( np.divide(warped_corners, warped_corners[2,:] ).T ) )

    # Trouver l'étendue des images déformée
    all_corners = np.vstack(all_corners)

    # La forme globale du cadre sera max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = corner_max - corner_min

    # Conversion en nombres entiers avec np.ceil et dtype
    output_shape = tuple( np.ceil(output_shape).astype(int) )
    
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
    raise NotImplementedError("TODO 3,4 : dans panorama.py non implémenté")
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
    raise NotImplementedError("TODO BONUS : dans panorama.py non implémenté")
    # TODO-BLOC-FIN

    return panorama