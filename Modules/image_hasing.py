import random
from collections import defaultdict
from itertools import combinations

import imagehash
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, adjusted_rand_score
import copy


class DuplicateHashEvaluator:
    """
    Classe pour détecter des doublons visuels avec différentes méthodes de hash,
    tester différents thresholds et tuner automatiquement pour maximiser F1-score
    (équilibre Precision / Recall).
    """

    def __init__(
        self, images_list, image_loader, labels_gt, hash_size=8, random_state=42
    ):
        self.images_list = images_list
        self.loader = image_loader
        self.labels_gt = labels_gt
        self.hash_size = hash_size
        self.random_state = random_state
        self.best_config = None
        self.best_score = None
        self.rng = random.Random(self.random_state)
        np.random.seed(self.random_state)

    def _compute_hash(self, img_path, method):
        """Génère le hash d'une image selon la méthode choisie."""
        img = self.loader(img_path).convert("RGB").resize((256, 256))
        if method == "phash":
            return imagehash.phash(img, hash_size=self.hash_size)
        if method == "ahash":
            return imagehash.average_hash(img, hash_size=self.hash_size)
        if method == "dhash":
            return imagehash.dhash(img, hash_size=self.hash_size)
        if method == "whash":
            return imagehash.whash(img, hash_size=self.hash_size)
        raise ValueError(f"Méthode inconnue {method}")

    @staticmethod
    def _union_find_clusters(hashes, threshold):
        """
        Regroupe des hashes perceptuels par distance de Hamming via
        union-find, et retourne pour chaque élément l'indice de sa racine.
        """
        n = len(hashes)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for i in range(n):
            for j in range(i + 1, n):
                if (hashes[i] - hashes[j]) <= threshold:
                    union(i, j)

        return [find(i) for i in range(n)]

    def _cluster(self, hashes, threshold):
        """Cluster les images selon la distance de Hamming."""
        roots = self._union_find_clusters(hashes, threshold)
        unique = {v: idx for idx, v in enumerate(set(roots))}
        return np.array([unique[r] for r in roots])

    def find_images_cluster(self, method, threshold):
        """Cluster les images selon la distance de Hamming."""
        hashes = [self._compute_hash(img, method) for img in self.images_list]
        roots = self._union_find_clusters(hashes, threshold)

        clusters_dict = defaultdict(list)
        for idx, root in enumerate(roots):
            clusters_dict[root].append(self.images_list[idx])

        return clusters_dict

    def _pairwise_metrics(self, labels_pred, sample_indices):
        """Calcule metrics + faux positifs/négatifs pour un échantillon."""
        n = len(labels_pred)
        fp = 0
        fn = 0

        gt_pairs = []
        pred_pairs = []

        for i, j in combinations(range(n), 2):
            idx_i = sample_indices[i]
            idx_j = sample_indices[j]

            gt_same = 1 if self.labels_gt[idx_i] == self.labels_gt[idx_j] else 0
            pred_same = 1 if labels_pred[i] == labels_pred[j] else 0

            gt_pairs.append(gt_same)
            pred_pairs.append(pred_same)

            if pred_same == 1 and gt_same == 0:
                fp += 1
            elif pred_same == 0 and gt_same == 1:
                fn += 1

        gt_pairs = np.array(gt_pairs)
        pred_pairs = np.array(pred_pairs)

        precision = precision_score(gt_pairs, pred_pairs, zero_division=0)
        recall = recall_score(gt_pairs, pred_pairs, zero_division=0)
        f1 = f1_score(gt_pairs, pred_pairs, zero_division=0)
        ari = adjusted_rand_score(
            [self.labels_gt[i] for i in sample_indices], labels_pred
        )

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ARI": ari,
            "FalsePositives": fp,
            "FalseNegatives": fn,
        }

    def evaluate(self, methods=None, thresholds=None, n_samples=5, sample_size=100):
        """
        Génère des échantillons, teste toutes les méthodes et thresholds,
        et retourne un DataFrame complet des métriques.
        Met à jour self.best_config pour la meilleure F1.
        """
        if methods is None:
            methods = ["phash", "ahash", "dhash", "whash"]
        if thresholds is None:
            thresholds = [3, 5, 7]

        results = []
        images_copy = copy.deepcopy(self.images_list)
        self.rng.shuffle(images_copy)  # shuffle reproducible

        best_f1 = -1
        best_config = None

        for s in range(n_samples):
            sample_imgs = images_copy[s * sample_size : (s + 1) * sample_size]
            sample_indices = [self.images_list.index(img) for img in sample_imgs]

            for method in methods:
                hashes = [self._compute_hash(img, method) for img in sample_imgs]

                for threshold in thresholds:
                    labels_pred = self._cluster(hashes, threshold)
                    metrics = self._pairwise_metrics(labels_pred, sample_indices)
                    metrics.update(
                        {
                            "sample": s,
                            "method": method,
                            "threshold": threshold,
                            "n_images": len(sample_imgs),
                        }
                    )
                    results.append(metrics)

                    # Mise à jour du meilleur F1
                    if metrics["F1"] > best_f1:
                        best_f1 = metrics["F1"]
                        best_config = {
                            "method": method,
                            "threshold": threshold,
                            "sample": s,
                            "metrics": metrics,
                        }

        self.best_config = best_config
        self.best_score = best_f1

        df_results = pd.DataFrame(results)
        return df_results

    def merge_dup_images(self, clusters_dict: dict, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fusionne les doublons détectés dans `data` en associant à chaque
        image une image représentative unique par cluster.

        Parameters
        ----------
        clusters_dict : dict
            Dictionnaire {cluster_id: [image_id, ...]} issu de
            `find_images_cluster`.
        data : pd.DataFrame
            DataFrame contenant une colonne `image_id`, à enrichir avec
            le cluster et l'image représentative de chaque doublon.

        Returns
        -------
        pd.DataFrame
            `data` enrichi des colonnes `cluster` et `unique_dup_img_name`.
        """
        rows = []
        for cluster_id, images in clusters_dict.items():
            unique_img = self.rng.choice(images)  # une image représentative par cluster
            for img in images:
                rows.append(
                    {
                        "image_id": img,
                        "cluster": cluster_id,
                        "unique_dup_img_name": unique_img,
                    }
                )

        temp_df = pd.DataFrame(rows)
        temp_df = temp_df.drop_duplicates(subset="image_id")
        data = data.merge(right=temp_df, on="image_id", how="left")
        return data

    def get_best(self):
        """Retourne la configuration qui maximise le F1-score."""
        return self.best_config
