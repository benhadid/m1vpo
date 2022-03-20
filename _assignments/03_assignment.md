---
type: assignment
date: 2022-04-28T10:00:00+1:00
title: 'Images panoramiques'
attachment: /static_files/assignments/hw2.zip
link: https://mybinder.org/v2/gh/benhadid/m1vpo/HEAD?labpath=static_files%2Fassignments%2Fhw2%2Fhw2.ipynb
weight: '30%'
hide_from_announcements: True
due_event:
    type: due
    date: 2022-05-15T17:00:00+1:00
    description: 'Devoir #3 - à remettre'
---

# Version en ligne

Vous pouvez cliquer sur cette image [![Binder](https://mybinder.org/badge_logo.svg)]({{page.link}}) pour lancer le devoir depuis un serveur en ligne (hébergé chez mybinder.org). Bien que cette méthode de lancement soit super simple, le temps de chargement du notebook et la qualité d'interaction dépendent largement du taux de charge des serveurs gratuits fournis par mybinder.org. 

# Version locale  

  1. Depuis votre machine VM, téléchargez le fichier de démarrage et décompressez le dans le répertoire de votre choix. 

  2. Depuis un terminal, accédez au répertoire du devoir (par exemple hw2) :
      ```bash
      user@vlab$ cd hw2
      ```      

  3. Lancez le bloc-notes jupyter à l'aide de la commande :   
      ```bash
      user@vlab$ jupyter notebook
      ```      

  4. Un navigateur Web devrait démarrer en affichant la liste des fichiers dans le répertoire courant. Cliquez sur le fichier associé au devoir (fichier avec l’extension `.ipynb`) pour afficher le bloc-notes.

  5. Lisez attentivement les questions et les commentaires dans le bloc-notes. Pour évaluer une réponse à une question du bloc-notes, exécutez la cellule (Cell \| Run Cells) associée à la question.

  6. Comme vous allez manipuler des images volumineuses dans ce devoir, il est possible que vous rencontriez des problèmes avec votre VM (machine devenant très lente ou ne répond plus aux commandes). Pour éviter cela, augmentez la taille mémoire allouée à votre VM depuis la console de VirtualBox.