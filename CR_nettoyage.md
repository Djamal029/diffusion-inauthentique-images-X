# Compte-rendu — Constitution et nettoyage du livrable de code (Groupe 21)

Ce document explique comment le contenu de `Livrable_Code_G21/` a été choisi à partir du
dépôt complet, en le confrontant à la méthodologie décrite dans `rapport/Materiel_&_Methode.Rmd`,
et liste précisément ce qui a été nettoyé. Les fichiers originaux à la racine du dépôt n'ont
pas été modifiés : ce dossier est une version sélectionnée et nettoyée, prête à être publiée.

## 1. Méthode de sélection

Pour chaque notebook à la racine, j'ai vérifié quels modules de `Modules/` il importe
réellement (`grep` sur les `from Modules... import ...`), puis j'ai confronté ces
imports aux étapes décrites dans le rapport (perceptual hashing → CLIP → UMAP → HDBSCAN →
optimisation bayésienne → réseau temporel pondéré → test de permutation → FDR → Louvain).
Cela a permis de distinguer :

- le pipeline final effectivement utilisé ;
- les brouillons/versions intermédiaires (notebooks non importés nulle part d'autre,
  ou clairement moins aboutis qu'une version voisine) ;
- les modules codés mais jamais branchés à un notebook.

## 2. Correspondance rapport → code retenu

| Section du rapport | Notebook(s) | Module(s) |
|---|---|---|
| 2.3 Description du jeu de données | `data_preprocessing.ipynb` | `data_preprocessing.py` |
| Analyse descriptive des comptes (README) | `desc-ech.ipynb`, `exploratory_analysis_f.ipynb` | `activity_analyzer.py`, `visual_analyser.py` |
| Analyse des interactions (README) | `exploratory_analysis_f.ipynb`, `stat_des_Ousseynou.ipynb` | `interaction_analyzer.py`, `interaction_analyzer_Ousseynou.py` |
| 2.4.1 Perceptual hashing | `hdbscan.ipynb` | `image_hasing.py` |
| 2.4.2 Embeddings CLIP | `hdbscan.ipynb`, `image_analysis.ipynb` | `image_analyzer.py`, `visual_analyser.py` |
| 2.4.3 UMAP | `hdbscan.ipynb` | `umap.py` |
| 2.4.4 HDBSCAN | `hdbscan.ipynb` | `hdbscan.py` |
| 2.4.5 Optimisation bayésienne | `hdbscan.ipynb` | `bayesian_optimization.py` |
| 2.5 Réseau de coordination, test de permutation, FDR, Louvain | `network-analysis.ipynb` | `temporal_coord.py`, `temporal_graph_analyzer.py` (voir §4) |
| (transverse) | — | `console_logger.py` (affichage utilisé par `hdbscan.py`) |

## 3. Notebooks exclus, et pourquoi

- **`exploratory_analysis.ipynb`** → écarté au profit de **`exploratory_analysis_f.ipynb`**.
  Les deux partagent le même titre et la même première cellule, mais la version `_f`
  contient 143 cellules contre 34 dans l'originale : c'est la version aboutie, l'autre
  est un brouillon antérieur resté dans le dépôt.
- **`image.ipynb`** → écarté au profit de **`image_analysis.ipynb`**. `image.ipynb` ne
  contient que 9 cellules quasiment vides ; `image_analysis.ipynb` contient l'analyse
  complète (63 cellules) et est le seul des deux réellement utilisé pour produire les
  résultats du rapport.

## 4. Modules exclus, et pourquoi

- **`network_analyzer.py`** → fichier vide (0 octet). Code mort, retiré.
- **`pca.py`** → n'est importé par aucun notebook du dépôt, et l'ACP n'apparaît pas
  dans la méthodologie du rapport (seul UMAP y est décrit pour la réduction de
  dimension). Vraisemblablement une piste explorée puis abandonnée. Non inclus, mais
  toujours disponible dans `Modules/` à la racine si vous souhaitez le réintégrer.

## 5. Point d'attention important : réseau temporel codé mais non branché

`temporal_coord.py` (`TemporalCoordinationAnalyzer`) et `temporal_graph_analyzer.py`
(`TemporalGraphAnalyzer`) implémentent **très fidèlement** la méthodologie statistique
centrale du rapport (section 2.5) : poids `w_uv = Σ e^{-λΔt}`, test de permutation,
correction FDR (Benjamini-Hochberg via `statsmodels`), extraction du sous-réseau
significatif, détection de communautés Louvain, comparaison à un réseau aléatoire.

Or **aucun notebook ne les importe**. En regardant `network-analysis.ipynb`, la même
logique (construction du réseau pondéré, Louvain, modularité réelle vs aléatoire) y est
réécrite **directement dans les cellules**, sous une forme moins structurée, plutôt que
d'appeler ces deux classes. Cela signifie deux choses :

1. Ces modules ne sont pas du code mort au sens strict — ils sont la version "propre" du
   cœur méthodologique du rapport — mais ils sont aujourd'hui **dupliqués** avec le code
   du notebook plutôt que réutilisés.
2. Je les ai inclus dans le livrable tels quels (avec un doublon de ligne corrigé dans
   `temporal_graph_analyzer.py`, cf. §6) car ils correspondent le mieux à la méthodologie
   décrite, mais **je n'ai pas réécrit `network-analysis.ipynb` pour qu'il les appelle** :
   c'est un choix qui change la logique du notebook et qui vous revient. Si vous voulez,
   je peux le faire dans un second temps.

## 6. Nettoyage effectué (comportement inchangé, sauf mention contraire)

**Notebooks** (`notebooks/`) :
- Sorties de cellules (outputs) supprimées sur les 7 notebooks retenus. Cela a fait
  passer `hdbscan.ipynb` de 29 Mo à 48 Ko et `network-analysis.ipynb` de 6,2 Mo à 74 Ko,
  et a supprimé des informations personnelles qui s'étaient glissées dans des messages
  d'erreur affichés (chemins locaux complets, dont le nom d'un membre du groupe dans un
  chemin Windows). Ces informations restent bien sûr dans les notebooks d'origine à la
  racine du dépôt, qui n'ont pas été modifiés.
- Remplacement du chemin personnel codé en dur `K:/Annee2/stats/Groupe21` (et, dans
  `hdbscan.ipynb`, `E:\ENSAI\2A\projet_stat\projet_stat_2a`) par `"."` : ce chemin
  pointe vers la racine du projet, où se trouvent déjà les dossiers `img/` et
  `source_img/`. Le notebook fonctionne donc pour n'importe qui clone le dépôt, sans
  dépendre d'un lecteur réseau personnel.
- Nom du kernel Jupyter (`pjst`) normalisé en `python3` dans les métadonnées, pour ne
  pas dépendre d'un environnement virtuel local nommé arbitrairement.

**`Modules/`** :
- `image_analyzer.py` : les deux mêmes chemins personnels codés en dur
  (`K:/Annee2/stats/Groupe21`, utilisés par défaut dans `display_specific_imgs` et
  `get_account_embeddings`) remplacés par un paramètre optionnel `source_path=None`
  résolu vers `"."` — même correction que pour les notebooks. Suppression de blocs de
  code mort en commentaire (mise en cache d'embeddings par hash, jamais activée) et
  correction d'une docstring qui documentait un paramètre `engagement_weighting`
  inexistant dans la signature.
- `image_hasing.py` : `_cluster` et `find_images_cluster` dupliquaient exactement la
  même logique d'union-find ; factorisée dans une seule méthode statique
  `_union_find_clusters`. Docstring de `merge_dup_images` (qui contenait encore le
  gabarit auto-généré `_summary_`/`_description_`, jamais rempli) réécrite. Suppression
  de lignes de code mort en commentaire.
- `hdbscan.py` : suppression de deux `print(n_neighbors)` de débogage laissés dans la
  fonction objectif de l'optimisation bayésienne. Remplacement du test
  `sil_score is not np.nan` par `not np.isnan(sil_score)`, plus robuste (le premier ne
  fonctionne que par coïncidence d'identité d'objet).
- `activity_analyzer.py` : suppression d'un `print(start_date)` de débogage, et de la
  méthode `plot_n_followers_by_n_posts` qui n'était qu'un `pass` (jamais implémentée,
  jamais appelée dans aucun notebook).
- `visual_analyser.py` : le `__init__` initialisait deux fois les mêmes attributs
  (`image_hashes`, `image_embeddings`, `dup_graph`, `sem_graph`) ; le premier bloc,
  redondant, a été supprimé. Suppression d'une ligne de code mort en commentaire et
  d'une ligne vide superflue.
- `interaction_analyzer_Ousseynou.py` : correction de fautes dans les messages
  d'erreur et commentaires (« abscente » → « absente », « condidère » → « considère »),
  suppression d'un bloc de code mort en commentaire (ancienne logique de sélection de
  colonne de date, remplacée depuis par une valeur fixe).
- `temporal_graph_analyzer.py` : suppression d'un appel dupliqué (copié-collé) à
  `G_random.community_multilevel(...)`, exécuté deux fois de suite pour rien.
- `umap.py` : suppression de deux commentaires de type note personnelle
  (`# pas trop utile mais bof`, `# fig.show()`), sans impact sur le comportement.
- `data_preprocessing.py`, `console_logger.py`, `bayesian_optimization.py`,
  `interaction_analyzer.py`, `temporal_coord.py` : déjà propres, copiés sans
  modification.

**`requirements.txt`** :
- Les lignes 27 à 32 du fichier original n'étaient pas des dépendances mais des
  commandes shell (`pip uninstall pywin32 -y`, `python -m pywin32_postinstall`, etc.)
  collées dans le fichier. Un `pip install -r requirements.txt` sur ce fichier échoue
  purement et simplement dès la ligne « pip uninstall... », qui n'est pas une syntaxe de
  requirement valide. Ces lignes ont été retirées ; les paquets concernés (`pywin32`,
  `easyocr`, `sentence-transformers`, `huggingface_hub`) ont été conservés comme
  dépendances normales.
- Ajout de `python-igraph` et `statsmodels`, utilisés par `temporal_coord.py` et
  `temporal_graph_analyzer.py` mais absents du fichier original.
- Retrait de `black` et `black[jupyter]` (outils de formatage de code, pas des
  dépendances nécessaires à l'exécution des analyses).

## 7. Ce qui n'a pas été touché

- Les données (`post_rehydrated.pickle`, `img/`, `source_img/`, fichiers CSV/pickle
  produits par les notebooks) ne sont pas dupliquées dans ce dossier : le livrable
  suppose qu'il est placé (ou que ses notebooks sont exécutés) à la racine du dépôt, là
  où se trouvent déjà ces données.
- Aucune logique métier n'a été modifiée : tous les changements ci-dessus sont soit des
  suppressions de code mort/debug, soit des corrections qui ne changent pas le résultat
  numérique des analyses (à l'exception du chemin d'images par défaut, qui ne
  fonctionnait de toute façon que sur le lecteur `K:` d'un des membres du groupe).

## 8. Structure du livrable

```
Livrable_Code_G21/
├── CR_nettoyage.md        (ce document)
├── README.md              (guide de démarrage rapide)
├── requirements.txt
├── Modules/
│   ├── data_preprocessing.py
│   ├── image_hasing.py
│   ├── image_analyzer.py
│   ├── umap.py
│   ├── hdbscan.py
│   ├── bayesian_optimization.py
│   ├── console_logger.py
│   ├── visual_analyser.py
│   ├── activity_analyzer.py
│   ├── interaction_analyzer.py
│   ├── interaction_analyzer_Ousseynou.py
│   ├── temporal_coord.py
│   └── temporal_graph_analyzer.py
└── notebooks/
    ├── data_preprocessing.ipynb
    ├── desc-ech.ipynb
    ├── exploratory_analysis_f.ipynb
    ├── image_analysis.ipynb
    ├── hdbscan.ipynb
    ├── network-analysis.ipynb
    └── stat_des_Ousseynou.ipynb
```
