# TP 1 - Segmentation

## Pose de germes

### Cadrillage image

1. Récupération de la partie entière du log en base 2 (largeur/hauteur)
  * Variables d1, d2 
2. Récupération de la valeur entière de la division de largeur/d1 hauteur/d2 
   * wCase/hCase = taille dimension

### Tirage aléatoire

1. Tirrage aléatoire de deux valeurs dans i € [0, d1-1] et j € [0, d2-1]
   * i et j sont les indices de la grille fragmenter 

## Croissance

### Frontière 

* 4-connexe -> R l'ensemble des points de R dont au moins un des 8-voisins n'est pas un élément de R

* 8-connexe -> R l'ensemble des points de R dont au moins un des 4-voisins n'est pas un élément de R

### Prédicats

## Fusion

## Multi-threading

## Source 

* [University of Nevada - region growing](https://www.cse.unr.edu/~bebis/CS791E/Notes/RegionGrowing.pdf)
* [OpenLayers - Region growing](https://openlayers.org/en/latest/examples/region-growing.html)
