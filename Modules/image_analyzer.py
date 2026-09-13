import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import requests
from io import BytesIO
from PIL import Image
import torch
import imagehash
import matplotlib.patches as patches


class ImageAnalyzer:
    """
    Analyse statistique et temporelle des images
    en tant qu'objets de diffusion sur X.

    Ici, on ne se concentre pas sur les colonnes source_
    parce que l'objectif principal est de regrouper et
    caractériser les images elles-mêmes, indépendamment
    de qui les a postées initialement.

    Les images sont considérées comme des identifiants
    de contenu (pas comme objets visuels).

    Aim
    ---
    Préparer des features quantitatives pour détecter
    les images potentiellement diffusées de manière
    inauthentique ou coordonnée.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame principal contenant les posts,
            comptes, dates et images.

        Aim
        ---
        Initialiser l'analyseur et préparer une colonne
        unique `image_id` pour toutes les images.
        """
        self.df = df.copy()

        # Colonnes image possibles
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

        # Normalisation => toujours une liste
        self.df["image_id"] = self.df[image_col].apply(
            lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else [])
        )

        # explode
        self.df = self.df.explode("image_id")

        # supprimer les lignes sans image
        self.df = self.df[self.df["image_id"].notna() & (self.df["image_id"] != "")]

    def image_popularity(self) -> pd.DataFrame:
        """
        Popularité globale des images.

        Returns
        -------
        pd.DataFrame
            image_id, n_posts, n_accounts, n_sources

        Aim
        ---
        Identifier quelles images sont les plus partagées,
        par combien de comptes différents et combien de
        comptes sources uniques.
        """
        return (
            self.df.groupby("image_id")
            .agg(
                n_posts=("post_id", "count"),
                n_accounts=("pf_account_id", "nunique"),
                n_sources=("source_pf_account_id", "nunique"),
            )
            .sort_values("n_posts", ascending=False)
            .reset_index()
        )

    def image_lifetime(self) -> pd.DataFrame:
        """
        Durée de vie temporelle des images.

        Returns
        -------
        pd.DataFrame
            image_id, first_seen, last_seen, lifetime_hours

        Aim
        ---
        Mesurer combien de temps une image circule sur X.
        Utile pour détecter des diffusions rapides et
        potentiellement coordonnées.
        (les images “virales” qui se répandent rapidement)
        """
        df = self.df.copy()
        df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True)

        out = (
            df.groupby("image_id")
            .agg(
                first_seen=("post_created_at", "min"),
                last_seen=("post_created_at", "max"),
            )
            .reset_index()
        )

        out["lifetime_hours"] = (
            out["last_seen"] - out["first_seen"]
        ).dt.total_seconds() / 3600

        return out.sort_values("lifetime_hours")

    def image_intensity(self) -> pd.DataFrame:
        """
        Intensité de rediffusion des images.

        Returns
        -------
        pd.DataFrame
            image_id, n_posts, n_accounts, posts_per_account

        Aim
        ---
        Détecter les images qui sont repostées de façon
        répétitive par un petit nombre de comptes,
        signe possible d'amplification artificielle.
        """
        df = (
            self.df.groupby("image_id")
            .agg(
                n_posts=("post_id", "count"),
                n_accounts=("pf_account_id", "nunique"),
            )
            .reset_index()
        )

        df["posts_per_account"] = df["n_posts"] / df["n_accounts"]

        return df.sort_values("posts_per_account", ascending=False)

    def image_bursts(self, timeframe: str = "1D") -> pd.DataFrame:
        """
        Détection de diffusions synchronisées d'images.

        Parameters
        ----------
        timeframe : str, optional
            Fenetre temporelle (ex: '1H', '6H', '1D').

        Returns
        -------
        pd.DataFrame
            image_id, time_bin, n_posts, n_accounts

        Aim
        ---
        Identifier les périodes où une image est postée
        de manière concentrée, pouvant indiquer
        une diffusion coordonnée.
        """
        df = self.df.copy()
        df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True)
        df["time_bin"] = df["post_created_at"].dt.floor(timeframe)

        return (
            df.groupby(["image_id", "time_bin"])
            .agg(
                n_posts=("post_id", "count"),
                n_accounts=("pf_account_id", "nunique"),
            )
            .reset_index()
            .sort_values("n_accounts", ascending=False)
        )

    def image_post_types(self) -> pd.DataFrame:
        """
        Répartition des types de posts par image.

        Returns
        -------
        pd.DataFrame
            image_id, n_original, n_quote, n_comment

        Aim
        ---
        Comprendre si une image est surtout postée
        en tant qu'original, quote ou commentaire,
        ce qui peut aider à détecter des diffusions atypiques.
        """
        return (
            self.df.groupby("image_id")
            .agg(
                n_original=("join_post_post_type", lambda x: (x == "original").sum()),
                n_quote=("join_post_post_type", lambda x: (x == "quote").sum()),
                n_comment=("join_post_post_type", lambda x: (x == "comment").sum()),
            )
            .reset_index()
        )

    def build_features(self) -> pd.DataFrame:
        """
        Construit un DataFrame de features pour toutes les images.

        Returns
        -------
        pd.DataFrame
            DataFrame avec une ligne par image et les colonnes :
            - n_posts : nombre total de posts
            - n_accounts : nombre de comptes distincts
            - n_sources : nombre de comptes sources distincts
            - lifetime_hours : durée de vie de l'image
            - posts_per_account : intensité de repost
            - n_original, n_quote, n_comment : répartition par type de post

        Aim
        ---
        Fournir un tableau complet de métriques par image pour :
        1) UMAP/HDBSCAN pour détecter des groupes ou anomalies
        2) Identifier les images diffusées de façon coordonnée
        3) Détecter des comportements inauthentiques
        """
        # Popularité
        pop = self.image_popularity()
        # Durée de vie
        lifetime = self.image_lifetime()[["image_id", "lifetime_hours"]]
        # Intensité de repost
        intensity = self.image_intensity()[["image_id", "posts_per_account"]]
        # Type de post
        post_types = self.image_post_types()

        # merge toutes les features
        features = pop.merge(lifetime, on="image_id", how="left")
        features = features.merge(intensity, on="image_id", how="left")
        features = features.merge(post_types, on="image_id", how="left")

        return features.sort_values("n_posts", ascending=False)

    def display_accounts_by_img_published(self, image_ids: list[str]) -> pd.DataFrame:
        """
        Affiche les comptes qui ont publié chaque image fournie.

        Parameters
        ----------
        image_ids : list[str]
            Liste d'IDs d'images à analyser.

        Returns
        -------
        pd.DataFrame
            DataFrame avec colonnes :
            - image_id : ID de l'image
            - pf_account_id : compte ayant publié l'image
            - n_posts : nombre de posts de ce compte pour cette image
            - first_post : date du premier post
            - last_post : date du dernier post
        """
        if not hasattr(self, "df") or "image_id" not in self.df.columns:
            raise ValueError("La colonne 'image_id' doit exister dans le DataFrame.")

        df = self.df[self.df["image_id"].isin(image_ids)].copy()
        df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True)

        result = (
            df.groupby(["image_id", "pf_account_id"])
            .agg(
                n_posts=("post_id", "count"),
                first_post=("post_created_at", "min"),
                last_post=("post_created_at", "max"),
            )
            .sort_values(["image_id", "n_posts"], ascending=[True, False])
        )

        return result

    def load_twitter_image(self, img_name: str):
        """
        Charge une image Twitter/X dans n'importe quel format :
        PNG, JPG, WEBP.

        Renvoie : tableau numpy lisible par plt.imshow()
        """

        # Formats possibles
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

        raise FileNotFoundError(
            f"Impossible de charger l'image Twitter : {img_name} "
            f"avec formats {formats}"
        )

    def display_specific_imgs(
        self,
        image_names: list[str],
        badges: list[int] | None = None,
        badge_color: str = "black",
        badge_text_color: str = "white",
        source_path: str | None = None,
        images_dir: list[str] = ["img", "source_img"],
        ncols: int = 5,
        figSize: tuple = (20, 20),
        from_internet: bool = False,
    ):
        """
        Affiche une grille d'images avec un badge numérique optionnel sur chaque image.

        Parameters
        ----------
        image_names : list[str]
            Liste des noms d'images à afficher
        badges : list[int] | None
            Liste des nombres à afficher dans les badges (même longueur que image_names)
        badge_color : str
            Couleur du fond du badge (ex: "black")
        badge_text_color : str
            Couleur du texte du badge
        source_path : str | None
            Dossier racine contenant les sous-dossiers d'images en local.
            Si None, utilise le dossier courant. Ignoré si `from_internet=True`.
        """
        # Vérifications
        if not isinstance(image_names, list) or len(image_names) == 0:
            raise ValueError("Vous devez fournir une liste non vide de noms d'images.")

        if badges is not None and len(badges) != len(image_names):
            raise ValueError("badges doit avoir la même longueur que image_names.")

        if not from_internet and len(images_dir) != 2:
            raise ValueError("images_dir doit contenir exactement deux sous-dossiers.")

        # Préparer les chemins
        if not from_internet:
            source_path = source_path or "."
            img_paths = [os.path.join(source_path, d, d) for d in images_dir]

            valid_paths = [p for p in img_paths if os.path.exists(p)]
            if not valid_paths:
                raise FileNotFoundError(
                    "Aucun des chemins images n'existe :\n" + "\n".join(img_paths)
                )

        # Préparer la grille
        n_images = len(image_names)
        nrows = (n_images + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figSize)
        axes = axes.flatten()

        # ------------------------------------------------------------------
        # Affichage des images + badges
        # ------------------------------------------------------------------
        for i, img_name in enumerate(image_names):

            try:
                # Charger l'image
                if not from_internet:
                    img = None
                    for path in img_paths:
                        candidate = os.path.join(path, img_name)
                        if os.path.exists(candidate):
                            img = mpimg.imread(candidate)
                            break
                    if img is None:
                        raise FileNotFoundError(f"{img_name} introuvable.")
                else:
                    img = self.load_twitter_image(img_name=img_name)

                # Affichage
                axes[i].imshow(img)
                axes[i].axis("off")

                # --------------------------------------------------------------
                # Ajout du badge (si fourni)
                # --------------------------------------------------------------
                if badges is not None:
                    count = badges[i]

                    # Paramètres du badge
                    badge_width = 0.22
                    badge_height = 0.13

                    # Rectangle arrondi
                    rect = patches.FancyBboxPatch(
                        (1 - badge_width - 0.03, 0.03),  # bas-droite
                        badge_width,
                        badge_height,
                        boxstyle="round,pad=0.2",
                        edgecolor="none",
                        facecolor=badge_color,
                        alpha=0.75,
                        transform=axes[i].transAxes,
                    )
                    axes[i].add_patch(rect)

                    # Texte du badge
                    axes[i].text(
                        1 - badge_width / 2 - 0.03,
                        0.03 + badge_height / 2,
                        str(count),
                        color=badge_text_color,
                        fontsize=14,
                        ha="center",
                        va="center",
                        weight="bold",
                        transform=axes[i].transAxes,
                    )

            except Exception as e:
                axes[i].text(0.1, 0.5, f"Erreur :\n{img_name}\n{str(e)}", color="red")
                axes[i].axis("off")

        # Désactiver axes restants
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    # semantic
    def get_account_embeddings(
        self,
        model,
        preprocess_func,
        account_id,
        device="cpu",
        from_internet=False,
        source_path: str | None = None,
        images_dir: list[str] = ["img", "src_image"],
    ):
        """
        Retourne l'embedding d'un compte basé sur toutes ses images.

        Parameters
        ----------
        model : modèle CLIP ou similaire
        preprocess_func : fonction de prétraitement des images
        account_id : int/str, id du compte
        device : "cpu" ou "cuda"
        from_internet : bool, charger depuis URL Twitter si True
        source_path : str | None
            Dossier racine contenant les sous-dossiers d'images en local.
            Si None, utilise le dossier courant. Ignoré si `from_internet=True`.
        images_dir : chemins des sous-dossiers d'images locales

        Returns
        -------
        torch.Tensor : vecteur unique du compte
        """

        # Récupérer toutes les images du compte
        df_account = self.df[self.df["pf_account_id"] == account_id]
        if df_account.empty:
            return torch.zeros(model.visual.output_dim)

        source_path = source_path or "."
        img_ids = df_account["image_id"].unique()
        embeddings = []

        for img_id in img_ids:
            try:
                # charger l'image
                if from_internet:
                    img = self.load_twitter_image(img_id)
                else:
                    img_path = None
                    for d in images_dir:
                        candidate = os.path.join(source_path, d, d, img_id)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    img = Image.open(img_path).convert("RGB")

                # prétraitement et embedding
                img_tensor = (
                    preprocess_func(img).unsqueeze(0).to(device)
                )  # ajoute le batch size
                with torch.no_grad():
                    emb = model.encode_image(img_tensor)
                emb = emb.squeeze(0).to(
                    device
                )  # utilise le device approprié (cpu ou gpu)

                embeddings.append(emb)

            except Exception as e:
                print(f"Erreur image {img_id}: {e}")
                continue

        if not embeddings:
            return torch.zeros(model.visual.output_dim)

        return torch.mean(torch.stack(embeddings), dim=0).squeeze()

    def get_image_hash(self, image_name):
        try:
            image_load = self.load_twitter_image(img_name=image_name)
            return imagehash.phash(image=image_load)
        except FileNotFoundError:
            print(f"Avertissement : image introuvable {image_name}")
            return None
        except Exception as e:
            print(f"Erreur lors du hash de l'image {image_name} : {e}")
            return None
