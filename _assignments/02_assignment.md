---
type: assignment
date: 2021-05-17T10:00:00+1:00
title: 'Détecteur, descripteur et mise en correspondance de primitives'
attachment: /static_files/assignments/hw1.zip
weight: '30%'
hide_from_announcements: True
due_event:
    type: due
    date: 2021-05-17T17:00:00+1:00
    description: 'Devoir #2 - à remettre'
---

# Démarche à suivre  

  1. Depuis votre machine VM, téléchargez le fichier de démarrage (fichier ```hw1.zip```) du devoir et décompressez le dans le répertoire de votre choix. 

  2. Depuis un terminal, accédez au répertoire du devoir (par exemple hw1) :
      ```bash
      user@vlab$ cd hw1
      ```      

  3. Lancez le bloc-notes jupyter à l'aide de la commande :   
      ```bash
      user@vlab$ jupyter notebook
      ```      

  4. Un navigateur Web devrait démarrer en affichant la liste des fichiers dans le répertoire courrant. Cliquez sur le fichier associé au devoir (fichier avec l’extension `.ipynb`) pour afficher le bloc-notes.

  5. Lisez attentivement les questions et les commentaires dans le bloc-notes. Pour évaluer une réponse à une question du bloc-notes, exécutez la cellule (Cell \| Run Cells) associée à la question.

  6. Pour pouvoir utiliser l’interface utilisateur **featuresUI.py** depuis la machine virtuelle, des paquetages supplémentaires devront être installés. En ce sens, depuis un terminal, tapez les commandes suivantes :
      ```bash
      user@vlab$ sudo apt update
      user@vlab$ sudo apt install python3-tk python-tk 
      ```      
      Le mot de passe pour le sudo est :  ```user```  si vous ne l’avez pas changé.
