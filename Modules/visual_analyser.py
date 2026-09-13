import torch
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class VisualAnalyzer:
    """
    Analyse visuelle et sémantique des images pour détecter :
    - duplications exactes (pHash)
    - similarité sémantique (CLIP embeddings)
    - clusters potentiellement coordonnés

    Utilise load_twitter_image pour récupérer les images depuis Twitter/X.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_loader=None,
        clip_model=None,
        preprocess=None,
        device="cpu",
    ):
        self.df = df.copy()
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.device = device

        # Loader par défaut : load_twitter_image
        if image_loader is None:
            self.image_loader = self.load_twitter_image
        else:
            self.image_loader = image_loader

        # dictionnaires pour stocker hashes et embeddings
        self.image_hashes = {}
        self.image_embeddings = {}

        # graphes
        self.dup_graph = nx.Graph()
        self.sem_graph = nx.Graph()
        self.sem_graph_knn = nx.Graph()
        self.image_cols = [
            "image_name",
            "source_image_name",
        ]
        self._prepare_images()

    def _prepare_images(self) -> None:
        """
        Normalise la colonne image pour l'analyse.

        - Crée une colonne `image_id` unique à partir
        des colonnes image disponibles.
        - Explose les listes d'images pour que chaque image
        occupe une ligne distincte.
        - Supprime les lignes sans image.
        """
        image_col = None
        for col in self.image_cols:
            if col in self.df.columns:
                image_col = col
                break

        if image_col is None:
            raise ValueError(
                "Aucune colonne image trouvée. Colonnes recherchées : "
                + ", ".join(self.image_cols)
            )

        self.df["image_id"] = self.df[image_col].apply(
            lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else [])
        )

        # explode
        self.df = self.df.explode("image_id")

        # supprimer les lignes sans image
        self.df = self.df[self.df["image_id"].notna() & (self.df["image_id"] != "")]

    # Loader par défaut
    def load_twitter_image(self, img_name: str):
        """
        Charge une image Twitter/X dans n'importe quel format : PNG, JPG, WEBP.
        Renvoie : objet PIL.Image ou None si l'image n'est pas trouvée.
        """
        formats = ["jpg", "png", "webp"]

        for fmt in formats:
            url = f"https://pbs.twimg.com/media/{img_name}?format={fmt}&name=large"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    return img
            except Exception:
                continue

        # Si aucun format n'a fonctionné
        print(f"Image introuvable ou inaccessible : {img_name}")
        return None

    # Calcul des embeddings CLIP
    def compute_clip_embeddings(self, column_: str = "unique_dup_img_name"):
        """
        Calcule les embeddings CLIP, skip les images introuvables.
        """
        if self.clip_model is None or self.preprocess is None:
            raise ValueError("clip_model et preprocess doivent être définis")

        for img_id in self.df[column_].unique():
            if img_id in self.image_embeddings:
                continue
            try:
                img = self.image_loader(img_id)  # peut lever FileNotFoundError
                img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.clip_model.encode_image(img_tensor)
                self.image_embeddings[img_id] = emb.squeeze(0).cpu()
            except FileNotFoundError:
                print(f"Image introuvable : {img_id}, skipping...")
                continue
            except Exception as e:
                print(f"Erreur embedding image {img_id}: {e}")
                continue

    def build_sem_graph(self, similarity_threshold=0.3):
        """
        Graphe basé sur la similarité sémantique CLIP
        """
        ids = list(self.image_embeddings.keys())
        self.sem_graph.add_nodes_from(ids)

        embeddings = np.stack([self.image_embeddings[i].numpy() for i in ids])
        embeddings_norm = normalize(embeddings, norm="l2", axis=1)
        sim_matrix = cosine_similarity(embeddings_norm)
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if sim_matrix[i, j] >= similarity_threshold:
                    self.sem_graph.add_edge(id1, id2, weight=sim_matrix[i, j])

    def build_sem_graph_knn(self, embeddings=None, k=10, sim_threshold=0.7):
        """
        Graphe sémantique KNN pondéré par cos similarity
        """

        # si embeddings non fournis, prendre self.image_embeddings
        if embeddings is None:
            ids = list(self.image_embeddings.keys())
            embeddings_array = np.stack([self.image_embeddings[i].numpy() for i in ids])
        else:
            embeddings_array = embeddings
            ids = list(range(len(embeddings_array)))

        # normalisation L2
        embeddings_norm = normalize(embeddings_array, axis=1)

        # KNN
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(ids)), metric="cosine")
        knn.fit(embeddings_norm)
        distances, indices = knn.kneighbors(embeddings_norm)

        # conversion distances -> cosine similarity
        sims = 1 - distances

        # construction graphe
        self.sem_graph_knn.add_nodes_from(ids)

        for i, img_id in enumerate(ids):
            for j_idx, sim in zip(indices[i][1:], sims[i][1:]):  # skip self
                neighbor_id = ids[j_idx]
                if sim >= sim_threshold:
                    self.sem_graph_knn.add_edge(img_id, neighbor_id, weight=float(sim))

        print(
            "Before deleting isolated nodes \n"
            f"Nodes: {self.sem_graph_knn.number_of_nodes()},"
            f" edges: {self.sem_graph_knn.number_of_edges()}"
        )

        isolated_nodes = [n for n, deg in self.sem_graph_knn.degree() if deg == 0]
        self.sem_graph_knn.remove_nodes_from(isolated_nodes)
        print(
            "After deleting isolated nodes \n"
            f"Nodes: {self.sem_graph_knn.number_of_nodes()},"
            f" edges: {self.sem_graph_knn.number_of_edges()}"
        )
        return self
