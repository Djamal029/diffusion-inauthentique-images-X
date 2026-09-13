import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from itertools import combinations


class TemporalGraphAnalyzer:
    def __init__(self, df, cluster_id, lambda_list, seed=42):
        """
        df : DataFrame contenant les données
        cluster_id : id du cluster à analyser
        lambda_list : liste de valeurs pour lambda_decay
        """
        self.df = df
        self.cluster_id = cluster_id
        self.lambda_list = lambda_list
        self.seed = seed
        random.seed(seed)

        # Stockage
        self.networkx_graphs = {}
        self.igraph_graphs = {}
        self.louvain_partitions = {}
        self.modularities = {}

    @staticmethod
    def _build_temporal_weighted_network(df, clusters_id, lambda_decay=0.3):
        df_filtered = df[df["cluster_hdbscan"].isin(clusters_id)].copy()
        df_filtered["post_created_at"] = pd.to_datetime(df_filtered["post_created_at"])

        G = nx.Graph()
        node_sizes = (
            df_filtered.groupby("pf_account_id")["unique_dup_img_name"].size().to_dict()
        )
        for node, size in node_sizes.items():
            G.add_node(node, size=size)

        for img_id, group in df_filtered.groupby("unique_dup_img_name"):
            group = group.sort_values("post_created_at")
            for row1, row2 in combinations(group.itertuples(), 2):
                acc1, acc2 = row1.pf_account_id, row2.pf_account_id
                delta_days = (
                    abs((row1.post_created_at - row2.post_created_at).total_seconds())
                    / 86400
                )
                weight = np.exp(-lambda_decay * delta_days)
                if acc1 != acc2:
                    if G.has_edge(acc1, acc2):
                        G[acc1][acc2]["weight"] += weight
                    else:
                        G.add_edge(acc1, acc2, weight=weight)
        return G

    @staticmethod
    def _random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def build_graphs_for_lambdas(self):
        """Construit les graphes pour toutes les valeurs de lambda."""
        for lamb in self.lambda_list:
            # Graphe NetworkX
            G_nx = self._build_temporal_weighted_network(
                self.df, [self.cluster_id], lamb
            )
            self.networkx_graphs[str(lamb)] = G_nx

            # Convertir en graphe igraph
            edges_with_weight = [
                (u, v, d["weight"]) for u, v, d in G_nx.edges(data=True)
            ]
            G_ig = ig.Graph.TupleList(
                edges_with_weight, weights=True, directed=False
            ).simplify(combine_edges="sum")

            # Calcul force des sommets
            G_ig.vs["strength"] = G_ig.strength(weights="weight")

            # Détection des communautés avec Louvain
            partition = G_ig.community_multilevel(weights=G_ig.es["weight"])
            self.louvain_partitions[str(lamb)] = partition
            mod_real = G_ig.modularity(partition.membership, weights=G_ig.es["weight"])

            # Graphe aléatoire pour comparaison
            G_random = G_ig.copy()
            G_random.rewire(n=200 * G_random.ecount())
            # Poids uniformes pour la comparaison au réseau aléatoire
            G_random.es["weight"] = [1] * G_random.ecount()
            partition_random = G_random.community_multilevel(
                weights=G_random.es["weight"]
            )
            mod_random = G_random.modularity(partition_random.membership)

            self.modularities[str(lamb)] = {
                "modularite_reel": mod_real,
                "modularite_random": mod_random,
            }

            # Attribution couleurs et communautés aux sommets
            community_colors = [self._random_color() for _ in range(len(partition))]
            G_ig.vs["community"] = [None] * len(G_ig.vs)
            G_ig.vs["color"] = [None] * len(G_ig.vs)
            for i, cluster in enumerate(partition):
                for vertex_id in cluster:
                    G_ig.vs[vertex_id]["community"] = i + 1
                    G_ig.vs[vertex_id]["color"] = community_colors[i]

            # Taille fixe des nœuds
            G_ig.vs["size"] = 7
            self.igraph_graphs[str(lamb)] = G_ig

    def export_graphs(self, folder_root="graphs_updated_lamb"):
        """Export GraphML pour tous les graphes construits."""
        folder_path = os.path.join(folder_root, f"cluster-{self.cluster_id}")
        os.makedirs(folder_path, exist_ok=True)

        for lamb, G_ig in self.igraph_graphs.items():
            file_path = os.path.join(
                folder_path, f"graph_cluster_{self.cluster_id}_lambda_{lamb}.graphml"
            )
            G_ig.write_graphml(file_path)

    def print_stats(self):
        """Affiche les statistiques des graphes."""
        for lamb, G_ig in self.igraph_graphs.items():
            print(f"Lambda = {lamb}")
            print("Modularité réel:", self.modularities[lamb]["modularite_reel"])
            print("Modularité aléatoire:", self.modularities[lamb]["modularite_random"])
            print("Density:", G_ig.density())
            print("Assortativity:", G_ig.assortativity_degree())
            print("Clustering coefficient:", G_ig.transitivity_avglocal_undirected())
            print("\n" + "-" * 10 + "\n")
