import pickle as pk
import pandas as pd
import os


class DataPreprocessing:
    """
    Classe de prétraitement des données X.

    Permet de charger les données et
    de normaliser les colonnes temporelles.
    """

    def __init__(self):
        self.data: pd.DataFrame | None = None

    def read_data(self, data_path: str, format_: str = "pickle") -> pd.DataFrame:
        """
        Charge les données depuis un fichier.

        Parameters
        ----------
        data_path : str
            Chemin vers le fichier.
        format_ : str, optional
            Format du fichier ('pickle', 'csv', etc.), par défaut 'pickle'.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            Si le format n'est pas supporté.
        """
        if not isinstance(data_path, str):
            raise TypeError(f"{data_path} doit être une chaine de caractères")

        if not os.path.exists(data_path):
            raise ValueError(f"Le chemin d'accès spéficié ({data_path}) n'existe pas.")

        if format_ == "pickle":
            with open(data_path, "rb") as f:
                self.data = pk.load(f)

        elif format_ == "csv":
            self.data = pd.read_csv(data_path)

        else:
            raise ValueError(
                f"Format '{format_}' non supporté. Utiliser 'pickle' ou 'csv'."
            )

        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Les données chargées ne sont pas un DataFrame pandas.")

        return self.data

    def parse_dates(self) -> None:
        """
        Convertit les colonnes de dates en datetime.

        Vérifie :
        - que la colonne existe
        - que le type initial est compatible (str, object, datetime)
        - lève une erreur si une colonne contient des types invalides

        Raises
        ------
        ValueError
            Si une colonne contient des types non convertibles.
        """
        if self.data is None:
            raise ValueError("Aucune donnée chargée. Appeler read_data() avant.")

        date_cols = [
            "source_date",
            "post_created_at",
            "source_account_registered_at",
            "account_registered_at",
        ]

        for col in date_cols:
            if col not in self.data.columns:
                continue

            # Vérification des types non null
            invalid_types = (
                self.data[col]
                .dropna()
                .apply(lambda x: not isinstance(x, (str, pd.Timestamp)))
            )

            if invalid_types.any():
                raise ValueError(
                    f"La colonne '{col}' contient des types non valides "
                    f"(attendu: str ou datetime)."
                )

            self.data[col] = pd.to_datetime(self.data[col], errors="raise", utc=True)
