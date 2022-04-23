---
type: assignment
date: 2022-03-31T10:00:00+1:00
title: 'Détecteur, descripteur et mise en correspondance de primitives'
attachment: /static_files/assignments/hw1.zip
link: https://colab.research.google.com/github/benhadid/m1vpo/blob/gh-pages/static_files/assignments/hw1/hw1.ipynb
weight: '30%'
hide_from_announcements: False
due_event:
    type: due
    date: 2022-04-17T17:00:00+1:00
    description: 'Devoir #2 - à remettre'
---

# Version en ligne

Vous pouvez cliquer sur cette image [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({{page.link}}) pour lancer le devoir depuis un serveur en ligne (Google Colaboratory). Bien que cette méthode de lancement soit super simple, il n'est pas possible de sauvegarder votre session de travail. Vous devez explicitement faire une copie du/des fichier-s modifié-s et coller le-s contenu-s dans des sessions futures.   

# Version locale 

  1. Depuis votre machine VM, téléchargez le fichier de démarrage et décompressez le dans le répertoire de votre choix. 

  2. Depuis un terminal, accédez au répertoire du devoir (par exemple hw1) :
      ```bash
      user@vlab$ cd hw1
      ```      

  3. Lancez le bloc-notes jupyter à l'aide de la commande :   
      ```bash
      user@vlab$ jupyter notebook
      ```      

  4. Un navigateur Web devrait démarrer en affichant la liste des fichiers dans le répertoire courant. Cliquez sur le fichier associé au devoir (fichier avec l’extension `.ipynb`) pour afficher le bloc-notes.

  5. Lisez attentivement les questions et les commentaires dans le bloc-notes. Pour évaluer une réponse à une question du bloc-notes, exécutez la cellule (Cell \| Run Cells) associée à la question.

  6. Pour pouvoir utiliser l’interface graphique **featuresUI.py** dans la machine virtuelle, vous devez installer des paquetages supplémentaires comme suit :
      ```bash
      user@vlab$ sudo apt update
      user@vlab$ sudo apt install python3-tk python-tk 
      ```      
      Le mot de passe pour le sudo est :  ``user`` si vous ne l’avez pas changé.
