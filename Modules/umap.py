import umap
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import normalize


class UmapDimensionReducer:
    """
    Réduction de dimension avec UMAP pour données étiquetées ou non.

    Notes
    -----
    Compatible avec des embeddings sous forme :

        df = pd.DataFrame(emb_matrix, index=img_ids)

    Les identifiants sont alors gérés via l'index.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        metric: str = "cosine",
        n_components: int = 10,
        random_state: int = 42,
        data: pd.DataFrame | None = None,
    ):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("`n_neighbors` doit être un entier strictement positif.")

        if not isinstance(min_dist, (int, float)) or not (0.0 <= min_dist <= 1.0):
            raise ValueError("`min_dist` doit être un réel compris entre 0 et 1.")

        self.n_neighbors = n_neighbors
        self.min_dist = float(min_dist)
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        self.data = data
        self.reducer = None
        self._X_reduced = None

    def reduce(self):
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state,
            min_dist=self.min_dist,
        )
        return self

    def fit_transform(self):
        """
        Applique UMAP sur les données.

        Retour
        ------
        X_reduced : numpy.ndarray
            Matrice réduite (n_samples x n_components).
        """
        if self.reducer is None:
            self.reduce()

        if self.data is None:
            raise ValueError("Aucune donnée fournie pour la réduction.")

        X = self.data.values  # on récupère emb_matrix
        X = normalize(X, norm="l2")

        self._X_reduced = self.reducer.fit_transform(X)
        return self._X_reduced

    def plot(
        self, dims: int = 2, labels: str | None = None, title: str = "Projection UMAP"
    ):
        """
        Visualise la projection UMAP avec Plotly.
        """
        if self._X_reduced is None:
            raise RuntimeError("Appeler `fit_transform` avant l'affichage.")

        if dims not in (2, 3):
            raise ValueError("`dims` doit être 2 ou 3.")

        df = pd.DataFrame(
            self._X_reduced,
            index=self.data.index,
            columns=[f"UMAP-{i+1}" for i in range(self.n_components)],
        )

        if labels is not None and labels in self.data.columns:
            df["labels"] = self.data[labels]
        else:
            labels = None

        if dims == 2:
            fig = px.scatter(
                df,
                x="UMAP-1",
                y="UMAP-2",
                color=labels,
                hover_name=df.index,
                opacity=0.7,
                title=title,
            )
        else:
            fig = px.scatter_3d(
                df,
                x="UMAP-1",
                y="UMAP-2",
                z="UMAP-3",
                color=labels,
                hover_name=df.index,
                opacity=0.7,
                title=title,
            )

        return fig
