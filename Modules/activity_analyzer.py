import pandas as pd
import matplotlib.pyplot as plt


class SourceActivityAnalyzer:
    """
    Analyse séparée de l'activité source
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def source_activity(
        self,
        source_account_ids: list[str] | None = None,
        period: tuple | None = None,
        post_type: list | None = None,
    ) -> pd.DataFrame:
        """
        Activité des comptes source (production de contenu).

        Parameters
        ----------
        source_account_ids : list[str], optional
            Si fourni, filtre l'activité sur ce compte source uniquement.
        period : tuple of (start_date, end_date), optional
            Filtre les posts de la source sur une période donnée.
        post_type: list od (original, comment, quote), optional

        Returns
        -------
        pd.DataFrame
            DataFrame avec colonnes :
            source_pf_account_id, source_account_name, n_posts,
            total_engagements, max_followers, max_following,
            followers_last_post, following_last_post, total_engagements_last_post
        """

        df = self.df.dropna(subset=["source_pf_account_id"]).copy()

        if post_type:
            df = df[df["join_post_post_type"].isin(post_type)]

        # Filtre sur source_account_id si fourni
        if source_account_ids:
            df = df[df["source_pf_account_id"].isin(source_account_ids)]

        # Filtre sur période si fourni
        if period:
            start_date, end_date = period
            df = df[
                (df["source_date"] >= pd.to_datetime(start_date, utc=True))
                & (df["source_date"] <= pd.to_datetime(end_date, utc=True))
            ]

        # le dernier post de chaque source
        last_posts = (
            df.sort_values("source_date").groupby("source_pf_account_id").tail(1)
        )

        # Agrégation
        result = df.groupby(
            ["source_pf_account_id", "source_account_name"], as_index=False
        ).agg(
            n_posts=("source_post_id", "nunique"),
            total_engagements=("source_post_engagements", "sum"),
            max_followers=("source_account_followers", "max"),
            max_following=("source_account_following", "max"),
        )

        # ajout des infos du dernier post
        result = result.merge(
            last_posts[
                [
                    "source_pf_account_id",
                    "source_account_followers",
                    "source_account_following",
                    "source_post_engagements",
                    "source_account_registered_at",
                    "source_date",
                ]
            ].rename(
                columns={
                    "source_account_followers": "followers_last_post",
                    "source_account_following": "following_last_post",
                    "source_post_engagements": "total_engagements_last_post",
                    "source_date": "last_post_date",
                }
            ),
            on="source_pf_account_id",
            how="left",
        )

        # Trier par nombre de posts
        result = result.sort_values("n_posts", ascending=False)

        return result

    def source_activity_over_time(
        self,
        source_account_ids: list[str] | None = None,
        period: tuple | None = None,
        post_type: list | None = None,
        freq: str = "D",  # 'D' = jour, 'W' = semaine, 'M' = mois
    ) -> pd.DataFrame:
        """
        Calcule la fréquence de publication des comptes source par jour,
        semaine ou mois, avec l'engagement moyen et le nombre moyen de followers.

        Parameters
        ----------
        source_account_ids : list[str] | None
            Filtrer sur ces comptes source.
        period : tuple | None
            Filtrer les posts dans la période (start_date, end_date)
        post_type : list | None
            Filtrer par type de post ('original', 'comment', 'quote')
        freq : str
            Fréquence pour l'agrégation :
            'D' = jour, 'W' = semaine, 'M' = mois

        Returns
        -------
        pd.DataFrame
            DataFrame avec colonnes :
            - source_pf_account_id
            - source_account_name
            - period_start : début de la période
            - n_posts : nombre moyen de posts par jour dans la période
            - total_engagements : engagements totaux dans la période
            - max_followers : followers maximum observés dans la période
        """
        df = self.df.dropna(subset=["source_pf_account_id"]).copy()
        df["source_date"] = pd.to_datetime(df["source_date"], utc=True)

        # Filtrage
        if period:
            start_date, end_date = period
            df = df[
                (df["source_date"] >= pd.to_datetime(start_date, utc=True))
                & (df["source_date"] <= pd.to_datetime(end_date, utc=True))
            ]

        if source_account_ids:
            df = df[df["source_pf_account_id"].isin(source_account_ids)]

        if post_type:
            df = df[df["join_post_post_type"].isin(post_type)]

        # Créer une colonne période pour l'agrégation
        df["period_start"] = df["source_date"].dt.to_period(freq).dt.start_time

        # Compter les posts par compte et période
        agg_df = (
            df.groupby(["source_pf_account_id", "source_account_name", "period_start"])
            .agg(
                n_posts_total=("source_post_id", "count"),
                total_engagements=("source_post_engagements", "sum"),
                max_followers=("source_account_followers", "max"),
            )
            .reset_index()
        )

        # Calculer la durée de chaque période en jours
        if freq == "D":
            agg_df["days_in_period"] = 1
        elif freq == "W":
            agg_df["days_in_period"] = 7
        elif freq == "M":
            # Nombre de jours dans le mois
            agg_df["days_in_period"] = agg_df["period_start"].dt.days_in_month
        else:
            raise ValueError("freq doit être 'D', 'W' ou 'M'")

        # Fréquence moyenne de publication par jour
        agg_df["n_posts"] = agg_df["n_posts_total"] / agg_df["days_in_period"]

        return agg_df[
            [
                "source_pf_account_id",
                "source_account_name",
                "period_start",
                "n_posts",
                "total_engagements",
                "max_followers",
            ]
        ].sort_values(["source_pf_account_id", "period_start"])

    def plot_source_activity_over_time(self, df: pd.DataFrame, value: str = "n_posts"):
        """
        Trace l'évolution de l'activité des comptes source dans le temps.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame renvoyé par `source_activity_over_time` avec colonnes :
            ['source_pf_account_id', 'source_account_name', 'period_start', value]
        value : str, optional
            La colonne à tracer : 'n_posts' ou 'total_engagements', par défaut 'n_posts'
        """
        plt.figure(figsize=(12, 6))

        for account_id, group in df.groupby("source_pf_account_id"):
            plt.plot(
                group["period_start"],
                group[value],
                marker="o",
                label=f"{group['source_account_name'].iloc[0]} ({account_id})",
            )

        plt.xlabel("Période")
        plt.ylabel(value)
        plt.title(f"Évolution de {value} par compte source")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
