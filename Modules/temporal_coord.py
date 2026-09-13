import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

from itertools import combinations
from collections import defaultdict
from statsmodels.stats.multitest import multipletests


class TemporalCoordinationAnalyzer:
    """
    Analyse statistique de coordination temporelle entre comptes.

    Cette classe implémente un pipeline complet :

    1. Construction d'un réseau temporel pondéré
    2. Test statistique par permutation
    3. Estimation de p-values empiriques
    4. Correction FDR
    5. Extraction du backbone significatif
    6. Analyse réseau (Louvain + métriques)
    7. Visualisations et export

    Required DataFrame columns
    --------------------------
    pf_account_id
    unique_dup_img_name
    post_created_at
    """

    def __init__(self, df, lambda_decay=0.3, B=1000, alpha=0.05, seed=42):

        self.df = df.copy()
        self.df["post_created_at"] = pd.to_datetime(self.df["post_created_at"])

        self.lambda_decay = lambda_decay
        self.B = B
        self.alpha = alpha

        np.random.seed(seed)

        self.results_df = None
        self.backbone = None
        self.network_analysis = None

    # -------------------------------------------------
    # TEMPORAL NETWORK
    # -------------------------------------------------

    def build_temporal_network(self, df=None):
        """
        Construit le réseau temporel pondéré.

        poids :
        w_ij = Σ exp(-λ Δt)

        Returns
        -------
        dict
            poids des arêtes
        """

        if df is None:
            df = self.df

        edge_weights = defaultdict(float)

        for img_id, group in df.groupby("unique_dup_img_name"):

            group = group.sort_values("post_created_at")
            rows = list(group.itertuples())

            for r1, r2 in combinations(rows, 2):

                if r1.pf_account_id == r2.pf_account_id:
                    continue

                delta = abs((r1.post_created_at - r2.post_created_at)) / np.timedelta64(
                    1, "D"
                )
                weight = np.exp(-self.lambda_decay * delta)

                key = tuple(sorted((r1.pf_account_id, r2.pf_account_id)))

                edge_weights[key] += weight

        return edge_weights

    # -------------------------------------------------
    # TIMESTAMP PERMUTATION
    # -------------------------------------------------

    def permute_timestamps(self):
        """
        Permutation des timestamps à l'intérieur de chaque image.

        Returns
        -------
        DataFrame
        """

        df_perm = self.df.copy()

        for img, group in self.df.groupby("unique_dup_img_name"):

            idx = group.index

            permuted = np.random.permutation(group["post_created_at"].values)

            df_perm.loc[idx, "post_created_at"] = pd.to_datetime(permuted, utc=True)

        return df_perm

    # -------------------------------------------------
    # PERMUTATION TEST
    # -------------------------------------------------

    def run_permutation_test(self):
        """
        Test statistique par permutation.

        Returns
        -------
        DataFrame
        """

        observed = self.build_temporal_network()

        permuted_weights = defaultdict(list)

        for b in range(self.B):

            if b % 100 == 0:
                print(f"Permutation {b}/{self.B}")

            df_perm = self.permute_timestamps()

            perm_net = self.build_temporal_network(df_perm)

            all_edges = set(observed.keys()) | set(perm_net.keys())

            for edge in all_edges:

                weight = perm_net.get(edge, 0.0)

                permuted_weights[edge].append(weight)

        results = []

        for edge, w_obs in observed.items():

            null_dist = permuted_weights[edge]

            mean_null = np.mean(null_dist)
            std_null = np.std(null_dist)

            z = (w_obs - mean_null) / std_null if std_null > 0 else 0

            null = np.array(null_dist)

            p = (1 + np.sum(null >= w_obs)) / (self.B + 1)

            results.append(
                {
                    "account_1": edge[0],
                    "account_2": edge[1],
                    "W_obs": w_obs,
                    "null_dist": null_dist,
                    "Z": z,
                    "p_value": p,
                }
            )

        self.results_df = pd.DataFrame(results)

        return self.results_df

    # -------------------------------------------------
    # FDR CORRECTION
    # -------------------------------------------------

    def apply_fdr(self):
        """
        Correction Benjamini-Hochberg.
        """

        rejected, p_adj, _, _ = multipletests(
            self.results_df["p_value"], alpha=self.alpha, method="fdr_bh"
        )

        self.results_df["p_adj"] = p_adj
        self.results_df["significant"] = rejected

        return self.results_df

    # -------------------------------------------------
    # BACKBONE NETWORK
    # -------------------------------------------------

    def build_backbone(self):
        """
        Construit le réseau significatif.
        """

        G = nx.Graph()

        sig = self.results_df[self.results_df["significant"]]

        for _, r in sig.iterrows():

            G.add_edge(
                r["account_1"],
                r["account_2"],
                weight=r["W_obs"],
                z=r["Z"],
                p_adj=r["p_adj"],
            )

        self.backbone = G

        return G

    # -------------------------------------------------
    # NETWORK ANALYSIS
    # -------------------------------------------------

    def analyze_network(self, n_random=50):
        """
        Analyse réseau avec Louvain.

        Inclut :

        - modularité réelle
        - modularité random via rewiring
        - densité
        - assortativité
        - betweenness
        """
        if not self.backbone:
            raise ValueError("Nothing to plot")
        edges = [
            (u, v, d.get("weight", 1)) for u, v, d in self.backbone.edges(data=True)
        ]

        G = ig.Graph.TupleList(edges, weights=True, directed=False)

        G = G.simplify(combine_edges="sum")

        partition = G.community_multilevel(weights="weight")

        modularity_real = G.modularity(partition, weights="weight")

        modularity_random = []

        for _ in range(n_random):

            G_rand = G.copy()

            G_rand.rewire(n=10 * G_rand.ecount())

            part_rand = G_rand.community_multilevel()

            modularity_random.append(G_rand.modularity(part_rand))

        density = G.density()

        assortativity = G.assortativity_degree()

        betweenness = G.betweenness(weights="weight")

        community_stats = []

        for i, comm in enumerate(partition):

            sub = G.subgraph(comm)

            community_stats.append(
                {"community_id": i, "size": len(comm), "density": sub.density()}
            )

        community_stats = pd.DataFrame(community_stats)

        G.vs["community"] = partition.membership
        G.vs["betweenness"] = betweenness

        self.network_analysis = {
            "igraph_graph": G,
            "modularity_real": modularity_real,
            "modularity_random_mean": np.mean(modularity_random),
            "density": density,
            "assortativity": assortativity,
            "community_stats": community_stats,
        }

        return self.network_analysis

    # -------------------------------------------------
    # MANHATTAN PLOT
    # -------------------------------------------------

    def plot_manhattan(self):

        df = self.results_df.sort_values("p_adj")

        df["neglog"] = -np.log10(df["p_adj"])

        plt.figure(figsize=(12, 6))

        plt.scatter(
            range(len(df)),
            df["neglog"],
            c=df["significant"].map({True: "green", False: "red"}),
            alpha=0.6,
        )

        plt.axhline(-np.log10(self.alpha), linestyle="--")

        plt.xlabel("Edge tests")
        plt.ylabel("-log10(p_adj)")

        plt.title("Manhattan plot")

        plt.show()

    # -------------------------------------------------
    # ERGODIC CONVERGENCE
    # -------------------------------------------------

    def plot_ergodic_convergence(self, edge_index=0):

        row = self.results_df.iloc[edge_index]

        null = np.array(row["null_dist"])

        indicators = (null >= row["W_obs"]).astype(int)

        p_hat = (1 + np.cumsum(indicators)) / (1 + np.arange(1, len(indicators) + 1))

        plt.figure(figsize=(8, 5))

        plt.plot(p_hat)

        plt.xlabel("Permutation")
        plt.ylabel("Estimated p-value")

        plt.title("Monte Carlo convergence")

        plt.show()

    # -------------------------------------------------
    # EXPORT
    # -------------------------------------------------

    def export_graphml(self, path):

        if self.network_analysis is None:
            raise ValueError("Run analyze_network() first")

        self.network_analysis["igraph_graph"].write_graphml(path)

    # -------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------

    def run_pipeline(self):

        print("Permutation test")

        self.run_permutation_test()

        print("FDR correction")

        self.apply_fdr()

        print("Building backbone")

        self.build_backbone()

        return self.results_df, self.backbone
