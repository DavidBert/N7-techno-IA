# Projet MODIA 2022

#### [Lien du github du projet](https://github.com/DavidBert/projet_MODIA)
#### [Lien du planning du projet](https://docs.google.com/spreadsheets/d/1ErpaE-O9dFBbZl_NJ2T71IaUyscCQI9AGAdsXX992jg/edit?usp=sharing)  


Dans ce projet, vous allez travailler sur des données<sup>[1](#myfootnote1)</sup>issues du site [Food.com](https://www.food.com/), un célèbre site de recettes de cuisine.   
![](https://github.com/DavidBert/projet_MODIA/raw/master/img/food.png)
Les données, disponibles [ici](https://drive.google.com/drive/folders/18JyoxTIrIH2s2wG6HtxGiKsdFtGSfUWm?usp=sharing), contiennent des informations sur des recettes de cuisines ainsi que des interactions de plusieurs utilisateurs avec les recettes.   

## Consignes:
Les parties 1, 2 et 3 sont à réaliser dans un même notebook.
### Partie 1: Recommandations simples:
Dans __un notebook__ présentez plusieurs stratégies de recommandation  de recettes:

*   Par popularité
*   Selon les étapes de la recette (colonne steps)
*   Selon la description de la recette 

Pour chacune de ces méthodes montrez quelques exemples des recommandation  obtenues.

### Partie 2: Analyse de sentiments:
Dans __un notebook__

*   À partir de la note donnée par les utilisateurs definissez une nouvelle variable sur le sentiment positif ou négatif d'un utilisateur vis-à-vis d'une recette.
Par exemple, toutes les notes inférieures à 3 sont négatives et celles supérieurs sont positives.  
Faites attention aux notes à 0 elles ne correspondent pas forcément à un sentiment négatif ou positif essayez d'en regarder quelques unes et décider de toutes les supprimer si elles posent problème.
*   En vous inspirant des parties NLP des TP [_Recommender_systems_](https://colab.research.google.com/github/DavidBert/N7-techno-IA/blob/master/code/recommender_systems/INSA_Reco_solution.ipynb#scrollTo=CSWUNjSB5oo-) et du TP [_Interpretability in Machine Learning_](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/interpretability/TP_interpretability_solution.ipynb), entrainez un modèle à prédire si un utilisateur a aimé ou non une recette à partir de son commentaire et utilisez la méthode LIME pour visualiser les mots permettant de justifier la décision de votre modèle.  
Montrez quelques exemples de prédiction et de visualisations des mots importants.  
Enregistrez votre modèle dans un fichier pickle.


### Partie 3: Neural Collaborative Filtering:
* Reprenez la classe ```NCF``` (Neural Collaborative Filtering )présente dans le [TP sur les systèmes de recommendations](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/recommender_systems/INSA_Reco_solution.ipynb) pour entrainer un modèle de Neural Collaborative Filtering à prédire les notes d'un utilisateur.
* Entrainez votre réseau sur les données train et testez le sur les données de test (calculez la Mean Absolute Error sur les données de test).
* Enregistrez les poids de votre réseau dans un fichier ```weight.pth```


### Partie 4 Scripts et Github:
*   Dans fichier ```model.py```, redefinissez la classe ```NCF``` (Neural Collaborative Filtering )présente dans le [TP sur les systèmes de recommandations](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/recommender_systems/INSA_Reco_solution.ipynb).
* Dans un fichier ```main.py```, implémentez un code permettant de prédire les notes d'un utilisateur.  
Ce programme ne fera pas d'entrainement mais récupérera les poids su réseau que vous aurez entraîné dans votre notebook.
Ce fichier sera exécuté comme un script et devra:
    * Récupérer les poids du réseau à partir d'un chemin donné en argument du job.
    * Récupérer le chemin d'un fichier de test contenant 10 interactions (```test_script.csv```) en arguments de la commande qui exécutera le script 
    *   Afficher les prédictions pour les 10 interactions du dataset de test.  

* Dans un fichier recommender_app.py , utilisez [gradio](https://gradio.app/) pour créer une application permettant de prédire si un commentaire, entré par l'utilisateur, est positif ou négatif (utilisez le composant [textbox](https://gradio.app/docs/#textbox)).

*   En bonus créez un dockerfile permettant de lancer le script ou l'application gradio depuis n'importe quelle machine.

Pour le livrable:

*   Faites un fork de ce repo Git.
*   Modifiez le pour qu'il contienne le notebook des parties 1,2 et 3.
*   Rajoutez les fichiers ```model.py```, ```main.py``` et ```recommender_app.py``` dans le repo Git.	
*   Modifiez le readme pour qu'il affiche vos noms et la commande permettant de lancer correctement le script ```main.py``` et l'application (ne mettez pas les fichiers de données dans le repo github!)
* Faites moi un pull request sur GitHub et envoyez moi un mail à david.bertoin@irt-saintexupery.com pour que je vérifie que tout est OK.
* Si vous avez fait le dockerfile, rajoutez dans le readme les commandes à exécuter pour le tester.

<a name="myfootnote1">1</a>: Les données ont été récoltées  pour l'article suivant:  
 [Generating Personalized Recipes from Historical User Preferences
Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley
EMNLP, 2019](https://www.aclweb.org/anthology/D19-1613/)