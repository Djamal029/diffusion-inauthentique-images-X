# Livrable de code — Groupe 21

Version sélectionnée et nettoyée du pipeline d'analyse (voir `CR_nettoyage.md` pour le
détail de ce qui a été gardé, écarté et corrigé, et pourquoi).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Ce dossier suppose qu'il est utilisé **à la racine du dépôt `projet_stat_2a`**, là où
se trouvent les données (`post_rehydrated.pickle`, `img/`, `source_img/`, etc.). Placez
`Modules/` et les notebooks de `notebooks/` à la racine du dépôt (ou ajustez les chemins
en tête de notebook) avant de les exécuter, dans l'ordre suivant :

1. `data_preprocessing.ipynb` — chargement et normalisation des données brutes.
2. `desc-ech.ipynb` / `exploratory_analysis_f.ipynb` / `stat_des_Ousseynou.ipynb` —
   statistiques descriptives sur les comptes et les interactions.
3. `image_analysis.ipynb` — statistiques et features au niveau image.
4. `hdbscan.ipynb` — détection des doublons (perceptual hashing), embeddings CLIP,
   réduction UMAP, clustering HDBSCAN et optimisation bayésienne des hyperparamètres.
5. `network-analysis.ipynb` — construction du réseau de coordination temporelle,
   test de permutation, correction FDR et détection de communautés (Louvain).

## Modules (`Modules/`)

| Module | Rôle |
|---|---|
| `data_preprocessing.py` | Chargement et normalisation des données (dates, formats) |
| `image_hasing.py` | Détection de doublons par perceptual hashing (phash/ahash/dhash/whash) |
| `image_analyzer.py` | Statistiques et features au niveau image |
| `visual_analyser.py` | Embeddings CLIP, graphes de similarité sémantique |
| `umap.py` | Réduction de dimension UMAP |
| `hdbscan.py` | Clustering HDBSCAN + tuning (grille et optimisation bayésienne) |
| `bayesian_optimization.py` | Optimisation bayésienne générique (wrapper `scikit-optimize`) |
| `console_logger.py` | Affichage console stylé, utilisé par `hdbscan.py` |
| `activity_analyzer.py` | Statistiques d'activité des comptes source |
| `interaction_analyzer.py` | Analyse des interactions / amplification |
| `interaction_analyzer_Ousseynou.py` | Variante de `interaction_analyzer.py` |
| `temporal_coord.py` | Réseau de coordination temporelle, test de permutation, FDR, Louvain |
| `temporal_graph_analyzer.py` | Sensibilité au paramètre λ (réseau + Louvain pour plusieurs λ) |

Voir `CR_nettoyage.md` pour la correspondance avec les sections du rapport.
