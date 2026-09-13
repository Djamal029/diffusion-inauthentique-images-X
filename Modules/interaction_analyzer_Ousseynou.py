import pandas as pd
import networkx as nx


class InteractionAnalyzer:
    """
    Analyse les interactions entre comptes et l'amplification autour des comptes source,
    avec possibilité de visualiser un réseau d'interactions avec NetworkX.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _apply_filters(
        self,
        pf_account_ids: list[str] | None = None,
        source_account_ids: list[str] | None = None,
        source_post_ids: list[str] | None = None,
        post_type: list | None = None,
        period: tuple | None = None,
        image_names: list[str] | None = None,
    ) -> pd.DataFrame:
        df = self.df.copy()

        if pf_account_ids:
            df = df[df["pf_account_id"].isin(pf_account_ids)]
        if source_account_ids:
            df = df[df["source_pf_account_id"].isin(source_account_ids)]
        if source_post_ids:
            df = df[df["source_post_id"].isin(source_post_ids)]
        if post_type:
            df = df[df["join_post_post_type"].isin(post_type)]
        if image_names:
            df = df[df["image_name"].isin(image_names)]
        if period:
            start, end = period
            start = pd.to_datetime(start, utc=True)
            end = pd.to_datetime(end, utc=True)
            date_col = (
                "post_created_at" if "post_created_at" in df.columns else "source_date"
            )
            df = df[(df[date_col] >= start) & (df[date_col] <= end)]
        return df

    def repeated_amplifiers(
        self,
        pf_account_ids: list[str] | None = None,
        source_account_ids: list[str] | None = None,
        source_post_ids: list[str] | None = None,
        post_type: list | None = None,
        period: tuple | None = None,
        image_names: list[str] | None = None,
        timeframe: str | None = None,  # fenêtre temporelle
    ) -> pd.DataFrame:
        """
        Détecte les amplificateurs répétitifs pour chaque publication,
        éventuellement dans une fenêtre temporelle.


        Cette méthode compte le nombre d'actions d'amplification (commentaire, citation)
        effectuées par chaque compte
        amplificateur sur un ou plusieurs comptes source.
        Elle peut regrouper les actions dans
        des fenêtres temporelles (`timeframe`) pour analyser
        les interactions coordonnées sur de courtes périodes.

        Parameters
        ----------
        pf_account_ids : list[str] | None, optional
            Liste des IDs de comptes amplificateurs à inclure.
            Si None, tous les comptes sont pris.
        source_account_ids : list[str] | None, optional
            Liste des IDs de comptes source à inclure.
            Si None, tous les comptes source sont pris.
        source_post_ids : list[str] | None, optional
            Liste des IDs de posts sources à inclure.
            Si None, tous les posts sont pris.
        post_type : list | None, optional
            Filtrer les interactions par type de post ('original', 'comment', 'quote').
            Si None, tous les types sont pris.
        period : tuple | None, optional
            Période à analyser sous la forme (start_date, end_date), dates incluses.
            Exemple : ('2024-01-01', '2025-09-15')
        image_names : list[str] | None, optional
            Filtrer les interactions par noms d'images associées aux posts.
        timeframe : str | None, optional
            Fenêtre temporelle pour regrouper les interactions, ex: '1D', '6H', '1H'.
            Si None, toutes les actions sont regroupées sur toute la période.

        Returns
        -------
        pd.DataFrame
            DataFrame contenant :
            - source_pf_account_id : ID du compte source
            - pf_account_id : ID du compte amplificateur
            - time_bin : tranche temporelle (si `timeframe` fourni)
            - interaction_count : nombre d'actions d'amplification observées

        Notes
        -----
        - Les fenêtres temporelles permettent d'identifier des amplificateurs coordonnés
        sur des périodes très courtes.
        - Si `timeframe` n'est pas fourni,
        toutes les interactions sont comptées globalement.
        """

        if "source_date" not in self.df.columns:
            raise ValueError("La colonne source_date est absente du jeu de données")

        df = self._apply_filters(
            pf_account_ids,
            source_account_ids,
            source_post_ids,
            post_type,
            period,
            image_names,
        )

        # On considère la date de publication du post source pour voir combien
        # de comptes interagissent avec la publication dans une fenêtre de temps
        date_col = "source_date"

        if timeframe:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            # Arrondir la date à la fenêtre choisie
            df["time_bin"] = df[date_col].dt.floor(timeframe)
            group_cols = ["source_pf_account_id", "pf_account_id", "time_bin"]
        else:
            group_cols = ["source_pf_account_id", "pf_account_id"]

        return (
            df.groupby(group_cols)
            .size()
            .reset_index(name="interaction_count")
            .sort_values("interaction_count", ascending=False)
        )

    def number_reaction(
        self,
        pf_account_ids: list[str] | None = None,
        source_post_ids: list[str] | None = None,
        post_type: list | None = None,
        period: tuple | None = None,
        image_names: list[str] | None = None,
        timeframe: str | None = None,  # fenêtre temporelle
    ) -> pd.DataFrame:
        """
        Détecter les comptes avec une forte activité,
        éventuellement dans une fenêtre temporelle.


        Cette méthode compte le nombre d'actions d'amplification (commentaire, citation)
        effectuées par chaque compte toutes publications confondues
        Elle peut regrouper les actions dans
        des fenêtres temporelles (`timeframe`) pour analyser
        les fortes activités sur de courtes périodes.

        Parameters
        ----------
        pf_account_ids : list[str] | None, optional
            Liste des IDs de comptes amplificateurs à inclure.
            Si None, tous les comptes sont pris.
        source_post_ids : list[str] | None, optional
            Liste des IDs de posts sources à inclure.
            Si None, tous les posts sont pris.
        post_type : list | None, optional
            Filtrer les interactions par type de post ('original', 'comment', 'quote').
            Si None, tous les types sont pris.
        period : tuple | None, optional
            Période à analyser sous la forme (start_date, end_date), dates incluses.
            Exemple : ('2024-01-01', '2025-09-15')
        image_names : list[str] | None, optional
            Filtrer les interactions par noms d'images associées aux posts.
        timeframe : str | None, optional
            Fenêtre temporelle pour regrouper les interactions, ex: '1D', '6H', '1H'.
            Si None, toutes les actions sont regroupées sur toute la période.

        Returns
        -------
        pd.DataFrame
            DataFrame contenant :
            - pf_account_id : ID du compte amplificateur
            - time_bin : tranche temporelle (si `timeframe` fourni)
            - reaction_count : nombre de réactions observées

        Notes
        -----
        - Les fenêtres temporelles permettent d'identifier des fortes activités
        suspectes sur des périodes très courtes.
        - Si `timeframe` n'est pas fourni,
        toutes les réactions sont comptées globalement.
        """

        if "post_created_at" not in self.df.columns:
            raise ValueError(
                "La colonne post_created_at est absente du jeu de données"
            )

        source_account_ids = None
        df = self._apply_filters(
            pf_account_ids,
            source_account_ids,
            source_post_ids,
            post_type,
            period,
            image_names,
        )

        # On considère la date de réaction à un poste pour voir le nombre d'actions
        # faites par le même compte dans la même fenêtre de temps
        date_col = "post_created_at"

        if timeframe:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            # Arrondir la date à la fenêtre choisie
            df["time_bin"] = df[date_col].dt.floor(timeframe)
            group_cols = ["pf_account_id", "time_bin"]
        else:
            group_cols = ["pf_account_id"]

        return (
            df.groupby(group_cols)
            .size()
            .reset_index(name="reaction_count")
            .sort_values("reaction_count", ascending=False)
        )

    def amplification_intensity(
        self,
        pf_account_ids: list[str] | None = None,
        source_account_ids: list[str] | None = None,
        source_post_ids: list[str] | None = None,
        period: tuple | None = None,
        post_type: list | None = None,
        image_names: list[str] | None = None,
        timeframe: str | None = None,  # fenêtre temporelle
    ) -> pd.DataFrame:
        """
        Mesure l'intensité d'amplification des comptes ou posts spécifiques.

        Cette fonction calcule, pour chaque
        compte source (et optionnellement pour chaque post) :
            - le nombre de comptes amplificateurs uniques (unique_amplifiers),
            - le nombre total d'actions d'amplification (total_actions),
            - et le ratio de duplication (duplication_ratio), qui indique combien
            d'actions ont été faites en moyenne par amplificateur unique.

        Parameters
        ----------
        pf_account_ids : list[str] | None
            Liste de comptes amplificateurs à inclure.
            Si None, tous les comptes sont inclus.
        source_account_ids : list[str] | None
            Liste de comptes source à inclure.
            Si None, tous les comptes sont inclus.
        source_post_ids : list[str] | None
            Liste de posts spécifiques à inclure.
            Si None, tous les posts sont inclus.
        period : tuple | None
            Période à filtrer sous forme (start_date, end_date).
            Si None, aucune restriction.
        post_type : list | None
            Filtre sur type de post ('original', 'comment', 'quote').
            Si None, tous les types sont inclus.
        image_names : list[str] | None
            Filtre sur les noms dimages.
            Si None, aucun filtre sur les images.

        Returns
        -------
        pd.DataFrame
            Un DataFrame avec les colonnes :
            - source_pf_account_id : identifiant du compte source
            - source_post_id : identifiant du post
            (présent uniquement si source_post_ids est fourni)
            - unique_amplifiers : nombre de comptes amplificateurs distincts
            - total_actions : nombre total d'actions d'amplification
            - duplication_ratio : ratio total_actions / unique_amplifiers,
            mesurant la répétitivité des amplifications

        Notes
        -----
        - La fonction trie le DataFrame final par `duplication_ratio` décroissant.
        - Elle utilise `_apply_filters` pour appliquer tous les filtres indiqués.
        """

        df = self._apply_filters(
            pf_account_ids,
            source_account_ids,
            source_post_ids,
            period,
            post_type,
            image_names,
        )

        # Colonnes de regroupement : compte source, éventuellement post
        group_cols = ["source_pf_account_id"]
        if "source_post_id" in df.columns and source_post_ids:
            group_cols.append("source_post_id")

        date_col = "source_date"
        if timeframe:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            # Arrondir la date à la fenêtre choisie
            df["time_bin"] = df[date_col].dt.floor(timeframe)
            group_cols.append("time_bin")

        # Agrégation
        agg = (
            df.groupby(group_cols)
            .agg(
                unique_amplifiers=("pf_account_id", "nunique"),
                total_actions=("pf_account_id", "count"),
            )
            .reset_index()
        )

        # Calcul du ratio de duplication
        agg["duplication_ratio"] = agg["total_actions"] / agg["unique_amplifiers"]

        # Trier par intensité
        return agg.sort_values("duplication_ratio", ascending=False)

    def build_interaction_network(self, interactions_df: pd.DataFrame) -> nx.DiGraph:
        """
        Construit un graphe orienté NetworkX :
        - Noeuds = comptes source et amplificateurs
        - Arêtes = actions d'amplification (poids = interaction_count)
        """

        G = nx.DiGraph()
        for _, row in interactions_df.iterrows():
            source = f"SOURCE_{row['source_pf_account_id']}"
            amp = f"AMP_{row['pf_account_id']}"
            G.add_node(source, type="source")
            G.add_node(amp, type="amplifier")
            G.add_edge(source, amp, weight=row["interaction_count"])
        return G
