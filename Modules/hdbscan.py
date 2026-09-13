import hdbscan
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from Modules.console_logger import DisplayMessage
from Modules.umap import UmapDimensionReducer
from Modules.bayesian_optimization import BayesianOptimization


class HdbscanClusterer:
    """
    Clusterisation avec HDBSCAN et tuning automatique des hyperparamètres,
    combiné avec une réduction de dimension UMAP.

    Notes
    -----
    Cette classe permet de :
    - réduire les embeddings via UMAP
        (nombre de composantes fixe, tunning de n_neighbors)
    - clusteriser avec HDBSCAN en testant différentes
        valeurs de min_cluster_size et min_samples
    - calculer les métriques : silhouette score
        (excluant le bruit), ratio de bruit et stabilité des clusters
    """

    def __init__(
        self,
        min_cluster_size=[5, 10, 15],
        min_sample_size=[5, 10, 20],
        metric="euclidean",
        cluster_selection_method="eom",
        random_state=42,
        min_dist=None,
        n_neighbors=None,
    ):
        """
        Initialisation du clusterer HDBSCAN.

        Paramètres
        ----------
        min_cluster_size : list[int], default=[5, 10, 15]
            Liste des tailles minimales de clusters HDBSCAN à tester.
        min_sample_size : list[int], default=[5, 10, 20]
            Liste des valeurs de min_samples à tester.
        metric : str, default='euclidean'
            Métrique de distance utilisée par HDBSCAN.
        cluster_selection_method : str, default='eom'
            Méthode de sélection des clusters ('eom' ou 'leaf').
        random_state : int, default=42
            Graine pour reproductibilité (uniquement utilisée pour UMAP).
        """
        self.min_cluster_size_list = min_cluster_size
        self.min_sample_size_list = min_sample_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        # attributs pour stocker le meilleur modèle
        self.best_model = None
        self.best_labels = None
        self.best_params = None
        self.best_scores = None

    def tune_umap_hdbscan(
        self,
        data: pd.DataFrame,
        number_of_neighbors_list=[10, 15, 20, 25, 30],
        n_comp=15,
    ):
        """
        Tuning combiné UMAP + HDBSCAN en fonction de la silhouette score.
        Calcule également le ratio de bruit et la stabilité des clusters.

        Paramètres
        ----------
        data : pd.DataFrame
            Matrice des embeddings (lignes = échantillons, colonnes = features).
        number_of_neighbors_list : list[int], optional, default=[10, 15, 20, 25, 30]
            Liste des valeurs de `n_neighbors` pour UMAP à tester.
        n_comp : int, optional, default=15
            Nombre de composantes pour UMAP (dimension réduite).

        Métriques calculées pour chaque combinaison :
        ---------------------------------------------
        - silhouette : silhouette score excluant les points bruités (-1)
        - noise_ratio : proportion de points considérés comme bruit
        - stability : moyenne de la persistance des clusters HDBSCAN

        Returns
        ------
        best_labels : np.ndarray
            Labels du meilleur clustering HDBSCAN selon la silhouette score.
            Les points bruités sont assignés à -1.

        Effets secondaires
        -----------------
        Met à jour l'objet avec :
        - self.best_model : meilleur clusterer HDBSCAN
        - self.best_labels : labels du meilleur clustering
        - self.best_params : dictionnaire avec n_neighbors,
                            min_cluster_size et min_samples
        - self.best_scores : dictionnaire avec silhouette, noise_ratio et stability
        """
        best_score = -np.inf
        best_model = None
        best_labels = None
        best_params = None
        best_metrics = None

        total_combinations = (
            len(number_of_neighbors_list)
            * len(self.min_cluster_size_list)
            * len(self.min_sample_size_list)
        )

        # progress bar
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )
        display_message = DisplayMessage()

        with progress:
            task = progress.add_task("Tuning UMAP+HDBSCAN", total=total_combinations)

            for n_neighbors in number_of_neighbors_list:
                # Réduction UMAP
                reducer = UmapDimensionReducer(
                    data=data, n_components=n_comp, n_neighbors=n_neighbors
                )
                X_umap = reducer.fit_transform()

                for min_cluster_size in self.min_cluster_size_list:
                    for min_samples in self.min_sample_size_list:
                        # Clusteriing HDBSCAN
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric=self.metric,
                            cluster_selection_method=self.cluster_selection_method,
                        )
                        labels = clusterer.fit_predict(X_umap)

                        # Ignorer si tout est bruit
                        if len(set(labels)) <= 1:
                            progress.update(task, advance=1)
                            continue

                        # Masque pour exclure le bruit
                        mask = labels != -1
                        if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                            sil_score = silhouette_score(X_umap[mask], labels[mask])
                        else:
                            sil_score = np.nan

                        # Ratio de bruit
                        noise_ratio = np.mean(labels == -1)

                        # Stabilité moyenne des clusters
                        if (
                            hasattr(clusterer, "cluster_persistence_")
                            and len(clusterer.cluster_persistence_) > 0
                        ):
                            stability = np.mean(clusterer.cluster_persistence_)
                        else:
                            stability = np.nan

                        # Choix du meilleur selon silhouette
                        if not np.isnan(sil_score) and sil_score > best_score:
                            best_score = sil_score
                            best_model = clusterer
                            best_labels = labels
                            best_params = {
                                "n_neighbors": n_neighbors,
                                "min_cluster_size": min_cluster_size,
                                "min_samples": min_samples,
                            }
                            best_metrics = {
                                "silhouette": sil_score,
                                "noise_ratio": noise_ratio,
                                "stability": stability,
                            }

                        progress.update(task, advance=1)
                        display_message.info(
                            message=(
                                f"Terminé: n_neighbors={n_neighbors}"
                                f", min_cluster_size={min_cluster_size}"
                                f", min_samples={min_samples}"
                            )
                        )

        # Stocker les meilleurs résultats
        self.best_model = best_model
        self.best_labels = best_labels
        self.best_params = best_params
        self.best_scores = best_metrics

        return best_labels

    def tune_umap_hdbscan_bopt_sil(
        self,
        data: pd.DataFrame,
        n_comp: int = 15,
        n_calls: int = 40,
        n_initial_points: int = 8,
        acq_func: str = "EI",
    ):
        """
        Bayesian optimization for UMAP + HDBSCAN using silhouette score.
        """

        space = [
            Integer(min(self.n_neighbors), max(self.n_neighbors), name="n_neighbors"),
            Real(0.0, self.min_dist, name="min_dist"),
            Integer(
                min(self.min_cluster_size_list),
                max(self.min_cluster_size_list),
                name="min_cluster_size",
            ),
            Integer(
                min(self.min_sample_size_list),
                max(self.min_sample_size_list),
                name="min_samples",
            ),
        ]

        @use_named_args(space)
        def objective(
            n_neighbors,
            min_dist,
            min_cluster_size,
            min_samples,
        ):
            n_neighbors = int(n_neighbors)
            min_cluster_size = int(min_cluster_size)
            min_samples = int(min_samples)
            # UMAP
            reducer = UmapDimensionReducer(
                data=data,
                n_components=n_comp,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=self.random_state,
            )
            X_umap = reducer.fit_transform()

            # HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=self.metric,
                cluster_selection_method=self.cluster_selection_method,
            )
            labels = clusterer.fit_predict(X_umap)

            # Reject degenerate solutions
            if len(set(labels)) <= 1:
                return 1.0  # bad score (we minimize)

            mask = labels != -1
            if np.sum(mask) <= 1 or len(set(labels[mask])) <= 1:
                return 1.0

            sil = silhouette_score(X_umap[mask], labels[mask])

            # Store best
            if self.best_scores is None or sil > self.best_scores["silhouette"]:
                self.best_model = clusterer
                self.best_labels = labels
                self.best_params = {
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                }
                self.best_scores = {
                    "silhouette": sil,
                    "noise_ratio": np.mean(labels == -1),
                    "stability": np.mean(clusterer.cluster_persistence_),
                }

            return -sil  # gp_minimize minimizes

        bopt = BayesianOptimization(
            objective_func=objective,
            space=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            random_state=self.random_state,
        )
        bopt.run_optimization()

        return bopt

    def get_results(self, index=None):
        """
        Retourne un DataFrame des labels du meilleur clustering.

        Paramètres
        ----------
        index : array-like, optional
            Identifiants des échantillons. Si None, utilise 0..n-1.

        Retour
        ------
        pd.DataFrame
            DataFrame avec colonnes :
            - 'id' : identifiant
            - 'cluster' : label HDBSCAN
        """
        if self.best_labels is None:
            raise RuntimeError(
                "Pas de clustering effectué. Appeler `tune_umap_hdbscan` avant."
            )

        if index is None:
            index = range(len(self.best_labels))

        return pd.DataFrame({"id": index, "cluster": self.best_labels})

    def summary(self):
        """
        Retourne un résumé du meilleur clustering.

        Retour
        ------
        dict
            - best_params : dictionnaire avec n_neighbors, min_cluster_size, min_samples
            - scores : dictionnaire avec silhouette, noise_ratio et stability
            - n_clusters : nombre de clusters détectés (excluant le bruit)
        """
        return {
            "best_params": self.best_params,
            "scores": self.best_scores,
            "n_clusters": len(set(self.best_labels))
            - (1 if -1 in self.best_labels else 0),
        }
