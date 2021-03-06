---
type: assignment
date: 2022-03-24T10:00:00+1:00
title: "Algèbre linéaire et manipulation d'images sous Python"
attachment: /static_files/assignments/hw0.zip
link: https://colab.research.google.com/github/benhadid/m1vpo/blob/gh-pages/static_files/assignments/hw0/hw0.ipynb
hide_from_announcements: False
due_event:
    type: due
    date: 2022-04-27T17:00:00+1:00
    description: 'Devoir #1 - à remettre'
---


# Tutoriels Python

Le langage Python sera utilisé pour la réalisation des devoirs de ce cours. En particulier, nous utiliserons Numpy pour le calcul scientifique. Si vous n'êtes pas familier avec Python et Numpy, les sites web suivants fournissent de très bons tutoriels pour eux.

 - [Numpy primer](https://www.cs.cornell.edu/courses/cs4670/2016sp/lectures/lec06_numpy.pdf) 
 - [Python Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)
 - [Official Numpy Quick Tutorial](https://docs.scipy.org/doc/numpy/user/quickstart.html)   
 - [Short Python Tutorial](https://www.tutorialspoint.com/python/)  
 - [Jupyter Notebook](https://realpython.com/jupyter-notebook-introduction/)


# Lancer le notebook associé au devoir 

## Version en ligne

Vous pouvez cliquer sur cette image [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({{page.link}}) pour lancer le devoir depuis un serveur en ligne (Google Colaboratory). Bien que cette méthode de lancement soit super simple, il n'est pas possible de sauvegarder votre session de travail. Vous devez explicitement faire une copie du/des fichier-s modifié-s et coller le-s contenu-s dans des sessions futures.   


## Version locale

Une machine virtuelle ubuntu semblable à la configuration utilisée dans les labos du département peut être téléchargée depuis la rubrique << Ressources >> de ce site. À l'intérieur de cette machine virtuelle, tous les paquetages et dépendances nécessaires ont été installés pour vous. Utilisez VirtualBox pour l'exécuter. En cas de besoin, le mot de passe de la machine virtuelle est : user. 

  1. Depuis votre machine VM, téléchargez le fichier de démarrage du devoir et décompressez le dans le répertoire de votre choix. 

  2. Depuis un terminal, accédez au répertoire du devoir (par exemple hw0) :
      ```bash
      user@vlab$ cd hw0
      ```      

  3. Lancez le bloc-notes jupyter à l'aide de la commande :   
      ```bash
      user@vlab$ jupyter notebook
      ```      

  4. Un navigateur Web devrait démarrer en affichant la liste des fichiers dans le répertoire courant. Cliquez sur le fichier associé au devoir (fichier avec l’extension `.ipynb`) pour afficher le bloc-notes.

  5. Lisez attentivement les questions et les commentaires dans le bloc-notes. Pour évaluer une réponse à une question du bloc-notes, exécutez la cellule (Cell \| Run Cells) associée à la question.
