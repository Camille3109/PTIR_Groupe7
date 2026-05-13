# Détection des modes de transport — NetMob 2025
**PTIR Groupe 7 — SARRAMEA Camille**

Détection automatique des modes de transport à partir de traces GPS, appliquée au dataset NetMob2025 (Paris et Île-de-France). Inspiré de Zheng et al. (2008).

---

## Vue d'ensemble

Le projet implémente **deux méthodes** de détection, chacune avec sa propre chaîne de fichiers. Le point d'entrée unique est `analyse.py`.

```
split.py
    └─► train_v1.py  ──► netmob_train.csv  ──► arbre_netmob_v1.py ──► post_processing.py ──► analyse.py
        train_v2.py  ──► netmob_train.csv  ──► arbre_netmob_v2.py ──┘
```

> Les deux méthodes partagent le **même fichier** `netmob_train.csv`. Il faut donc relancer le bon script `train_vX.py` avant de changer de méthode.

---

## Fichiers

### `split.py`
À lancer **une seule fois** au début du projet. Sépare les utilisateurs en deux groupes (70% train / 30% test) et sauvegarde les listes dans `train_users.json` et `test_users.json`.

---

### `nettoyer.py`
Nettoyage des fichiers GPS bruts. Appelé automatiquement par `classification_segments_v1.py` et `v2.py` — pas besoin de le lancer manuellement.

Pipeline en 3 étapes :
1. Suppression des coordonnées invalides (NaN, hors plage, hors IDF)
2. Détection et suppression des sauts aberrants (pics de vitesse isolés)
3. Fusion des points trop proches (distance < 10 m ET écart < 30 s)

Les fichiers nettoyés sont sauvegardés dans `fichiers_nettoyes/`.

---

### `classification_segments_v1.py` — Méthode 1
Segmentation **marche / non-marche** : chaque segment est classifié comme "marche" ou "autre" selon la vitesse et l'accélération, puis sous-segmenté si besoin.

Features calculées par segment (12 au total) :
- HCR (Heading Change Rate), SR (Stop Rate), VCR (Velocity Change Rate)
- Vitesse max, médiane, P99
- Accélération max
- % de temps à vitesse rapide (>40 km/h) et très rapide (>58 km/h)
- Durée et longueur du segment

Gère aussi les **trajets extra** : les points GPS hors des fenêtres temporelles déclarées sont regroupés en trajets supplémentaires (coupure à 10 min de gap ou après détection de stay-points).

---

### `classification_segments_v2.py` — Méthode 2 (plus proche du papier)
Segmentation par **variations de vitesse** : les changements de mode sont détectés via les ruptures dans le profil de vitesse.

Features calculées (5 seulement, comme dans l'article original) :
- HCR, SR, VCR, vitesse max, accélération max

---

### `train_v1.py` / `train_v2.py`
Génèrent le fichier `netmob_train.csv` à partir des utilisateurs du set d'entraînement.

Pour chaque utilisateur, les features GPS sont calculées et fusionnées avec les modes déclarés (`Mode_1` du dataset displacements). Le résultat est normalisé selon le mapping suivant :

| Mode brut | Mode normalisé |
|---|---|
| PRIV_CAR_DRIVER, PRIV_CAR_PASSENGER, TWO_WHEELER, ELECT_SCOOTER | CAR |
| TRAIN, TRAIN_EXPRESS | TRAIN |
| SUBWAY | SUBWAY |
| TRAMWAY | TRAMWAY |
| WALKING | WALK |
| BUS | BUS |
| BIKE, ELECT_BIKE | CYCLING |

> La catégorie `OTHER` du dataset n'est pas incluse : elle regroupe des modes trop hétérogènes sans signature GPS distinctive, et le modèle avait tendance à tout y classer dès qu'il n'était pas suffisamment confiant.

Les deux scripts utilisent le **multiprocessing** (N-1 cœurs) pour paralléliser le traitement.

---

### `arbre_netmob_v1.py` / `arbre_netmob_v2.py`
Chargent `netmob_train.csv`, entraînent un `RandomForestClassifier` **une seule fois au démarrage** (variable globale `GLOBAL_CLF`), puis l'appliquent à un fichier GPS utilisateur.

Paramètres du RandomForest : 150 arbres, profondeur max 10, `min_samples_leaf=5`, `max_samples=0.8`.

Sortie : un DataFrame `df_res` avec une ligne par segment, contenant les probabilités par mode, le mode prédit, la confiance, le `trip_id`, les coordonnées GPS, et les timestamps de début et fin du segment (`TIMESTAMP`, `TIMESTAMP_FIN`).

> `arbre_netmob_v1` doit être utilisé avec un `netmob_train.csv` généré par `train_v1.py`, et `arbre_netmob_v2` avec un CSV généré par `train_v2.py`. Mélanger les deux provoque une erreur `not in index`.

---

### `post_processing.py`
Post-traitement des prédictions brutes du RandomForest. Importé par `analyse.py` via `lancement_user()`.

Étapes appliquées à chaque utilisateur :
1. **Graphe de transition** : correction des prédictions peu confiantes en tenant compte de la cohérence séquentielle des modes (matrice de transition apprise sur le set d'entraînement)
2. **Vote glissant** (`sliding_majority_vote`, fenêtre=5) : lissage local des prédictions sur chaque trajet séparément
3. **Fusion de segments** (`fusion_segments`) : supprime les segments isolés incohérents (A→B→A devient A→A→A)
4. **Normalisation** : homogénéisation des noms de modes (ex: `metro` → `SUBWAY`, `taxi` → `CAR`)

Contient aussi :
- `calculer_duree_par_mode` : calcule le % de temps passé dans chaque mode via les timestamps GPS
- `comparer_predictions` : compare les modes prédits avec les modes déclarés (précision, rappel, F1 par trajet)
- `obtenir_infos_individu` : récupère sexe, âge et diplôme depuis `individuals_dataset.csv`

> Un post-processing spatial (OPTICS + KDTree + formule de Bayes) est aussi implémenté (`spatial_graph_post_processing`) mais donne de moins bons résultats et n'est pas utilisé par défaut.

---

### `analyse.py`
Point d'entrée principal. Lance l'analyse sur tous les utilisateurs du set de test en **parallèle** (multiprocessing, N-1 cœurs).

Pour chaque utilisateur :
- Appelle `lancement_user` (post_processing)
- Extrait les points de changement de mode (position GPS au moment d'une transition)
- Récupère les trajets déclarés et extra avec leurs dates, heures et modes prédits

Produit en sortie :

| Fichier | Contenu |
|---|---|
| `trajets_par_utilisateur.csv` | Un trajet par ligne (déclarés + extra), avec dates, modes prédits/réels, métriques |
| `points_changement.csv` | Coordonnées GPS de chaque transition de mode |
| `points_changement_map.html` | Carte interactive des transitions (couleur par type de transition) |
| `heatmap_transitions.html` | Carte de densité des zones de changement de mode |
| `analyse_transitions.png` | Histogramme + matrice des transitions observées |
| `analyse_demographique_transport.png` | Répartition modale par sexe, âge et diplôme |

---

## Ordre d'exécution

```
1. split.py                    # une seule fois
2. train_v1.py  (ou train_v2.py)   # génère netmob_train.csv
3. analyse.py                  # lance l'analyse complète
```

---

## Dépendances

```
pandas, numpy, scikit-learn, scipy, folium, matplotlib
```

---

## Notes importantes

- Les chemins sont codés en dur pour Windows (`C:\Users\Camille\...`). À adapter si changement de machine.
- `analyse.py` utilise `arbre_netmob_v1` (Méthode 1) par défaut. Pour passer à la Méthode 2, changer l'import dans `post_processing.py` (`from arbre_netmob_v2 import arbre`) et dans `analyse.py` ligne 80 et relancer `train_v2.py` au préalable.
- Le multiprocessing nécessite le bloc `if __name__ == "__main__":` dans `analyse.py` — obligatoire sur Windows.
