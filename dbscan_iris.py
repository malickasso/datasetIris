#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN Clustering - Partie 2: Pratique avec Python
Analyse complète avec dataset Iris
Dataset différent pour une nouvelle analyse
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')


class DBSCANAnalysis:
    """Classe pour l'analyse DBSCAN complète"""
    
    def __init__(self):
        self.data = None
        self.X = None
        self.X_scaled = None
        self.labels = None
        self.dbscan_model = None
        self.feature_names = None
        self.target_names = None
        
    def load_and_prepare_data(self):
        """1. Présentation du dataset"""
        print("\n" + "="*70)
        print("1. PRÉSENTATION DU DATASET")
        print("="*70)
        
        # Charger le dataset Iris
        iris = load_iris()
        self.X = iris.data
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        y_true = iris.target
        
        # Créer DataFrame pour analyse
        self.data = pd.DataFrame(self.X, columns=self.feature_names)
        self.data['true_class'] = y_true
        self.data['species'] = [self.target_names[i] for i in y_true]
        
        print(f"\n Dataset Iris chargé avec succès")
        print(f"  - Nombre d'échantillons: {self.X.shape[0]}")
        print(f"  - Nombre de caractéristiques: {self.X.shape[1]}")
        print(f"  - Classes réelles: {len(self.target_names)} ({', '.join(self.target_names)})")
        
        print("\ Description du dataset:")
        print("   Le dataset Iris contient des mesures de fleurs d'iris.")
        print("   Il s'agit d'un dataset classique en machine learning.")
        
        print("\ Aperçu des données:")
        print(self.data.head(10))
        
        print("\ Statistiques descriptives:")
        print(self.data[self.feature_names].describe())
        
        print("\ Distribution par espèce:")
        print(self.data['species'].value_counts())
        
        return self.data
    
    def preprocess_data(self):
        """2. Prétraitement des données"""
        print("\n" + "="*70)
        print("2. PRÉTRAITEMENT DES DONNÉES")
        print("="*70)
        
        print("\ Statistiques AVANT normalisation:")
        print(self.data[self.feature_names].describe().loc[['mean', 'std']])
        
        # Normalisation
        print("\  Normalisation avec StandardScaler...")
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print(f"\n Prétraitement terminé")
        print(f"  - Toutes les caractéristiques sont maintenant centrées (moyenne=0)")
        print(f"  - Toutes les caractéristiques ont un écart-type de 1")
        
        print("\ Statistiques APRÈS normalisation:")
        data_scaled = pd.DataFrame(self.X_scaled, columns=self.feature_names)
        print(data_scaled.describe().loc[['mean', 'std']])
        
        return self.X_scaled
    
    def find_optimal_epsilon(self):
        """Trouver epsilon optimal avec la méthode k-distance"""
        print("\n" + "="*70)
        print("3. DÉTERMINATION DES PARAMÈTRES")
        print("="*70)
        
        print("\n Recherche d'epsilon optimal avec la méthode k-distance...")
        
        # Calculer les k-distances (k=4 par défaut)
        k = 4
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X_scaled)
        distances, indices = neighbors.kneighbors(self.X_scaled)
        
        # Trier les distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Créer graphique k-distance
        plt.figure(figsize=(12, 6))
        plt.plot(distances, linewidth=2, color='steelblue')
        plt.ylabel('k-distance (k=4)', fontsize=12, fontweight='bold')
        plt.xlabel('Points triés par distance', fontsize=12, fontweight='bold')
        plt.title('Méthode k-distance pour déterminer Epsilon (Dataset Iris)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, 
                   label='Epsilon suggéré = 0.5')
        plt.legend(fontsize=11)
        plt.tight_layout()
        print(" Graphique affiché: iris_k_distance_plot.png")
        plt.show()
        
        # Epsilon suggéré (observé sur le graphique du coude)
        epsilon_optimal = 0.5
        
        print(f"\n✓ Paramètres sélectionnés:")
        print(f"  - Epsilon (eps): {epsilon_optimal}")
        print(f"  - Min_samples: {k}")
        print(f"  - Justification: k={k} (basé sur la dimensionnalité du dataset)")
        
        return epsilon_optimal
    
    def apply_dbscan(self, eps=0.5, min_samples=4):
        """4. Application du modèle DBSCAN"""
        print("\n" + "="*70)
        print("4. APPLICATION DE DBSCAN")
        print("="*70)
        
        print(f"\n Configuration du modèle:")
        print(f"  - eps = {eps}")
        print(f"  - min_samples = {min_samples}")
        
        # Créer et ajuster le modèle
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.dbscan_model.fit_predict(self.X_scaled)
        
        # Statistiques
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        
        # Distribution des clusters
        unique, counts = np.unique(self.labels, return_counts=True)
        
        print(f"\n Clustering terminé!")
        print(f"\n Résultats:")
        print(f"  - Nombre de clusters détectés: {n_clusters}")
        print(f"  - Points de bruit (outliers): {n_noise} ({n_noise/len(self.labels)*100:.1f}%)")
        
        print(f"\n Distribution des points:")
        for label, count in zip(unique, counts):
            if label == -1:
                print(f"  - Bruit: {count} points")
            else:
                print(f"  - Cluster {label}: {count} points")
        
        return self.labels
    
    def evaluate_and_visualize(self):
        """5. Évaluation et visualisation des résultats"""
        print("\n" + "="*70)
        print("5. ÉVALUATION ET VISUALISATION")
        print("="*70)
        
        # Calculer les métriques (exclure le bruit)
        mask = self.labels != -1
        X_no_noise = self.X_scaled[mask]
        labels_no_noise = self.labels[mask]
        
        print("\n Métriques de qualité du clustering:")
        
        if len(set(labels_no_noise)) > 1:
            silhouette = silhouette_score(X_no_noise, labels_no_noise)
            davies_bouldin = davies_bouldin_score(X_no_noise, labels_no_noise)
            calinski_harabasz = calinski_harabasz_score(X_no_noise, labels_no_noise)
            
            print(f"  - Silhouette Score: {silhouette:.3f} (plus proche de 1 = meilleur)")
            print(f"  - Davies-Bouldin Index: {davies_bouldin:.3f} (plus faible = meilleur)")
            print(f"  - Calinski-Harabasz Score: {calinski_harabasz:.2f} (plus élevé = meilleur)")
        else:
            print("  Pas assez de clusters pour calculer les métriques")
        
        # Visualisation 2D - Utiliser les 2 caractéristiques les plus importantes
        # Sepal length vs Sepal width et Petal length vs Petal width
        
        fig = plt.figure(figsize=(18, 6))
        
        # Subplot 1: DBSCAN - Sepal
        ax1 = plt.subplot(1, 3, 1)
        unique_labels = set(self.labels)
        colors_list = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors_list):
            if k == -1:
                col = 'black'
            
            class_member_mask = (self.labels == k)
            xy = self.X[class_member_mask, :2]  # Sepal features
            
            if k == -1:
                ax1.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=col,
                        markeredgecolor='k', markersize=8, alpha=0.8, 
                        label='Bruit', linewidth=2)
            else:
                ax1.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=10, alpha=0.8,
                        label=f'Cluster {k}')
        
        ax1.set_title('DBSCAN - Caractéristiques Sepal', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Sepal Length (cm)', fontsize=11)
        ax1.set_ylabel('Sepal Width (cm)', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: DBSCAN - Petal
        ax2 = plt.subplot(1, 3, 2)
        for k, col in zip(unique_labels, colors_list):
            if k == -1:
                col = 'black'
            
            class_member_mask = (self.labels == k)
            xy = self.X[class_member_mask, 2:]  # Petal features
            
            if k == -1:
                ax2.plot(xy[:, 0], xy[:, 1], 'x', markerfacecolor=col,
                        markeredgecolor='k', markersize=8, alpha=0.8, 
                        label='Bruit', linewidth=2)
            else:
                ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=10, alpha=0.8,
                        label=f'Cluster {k}')
        
        ax2.set_title('DBSCAN - Caractéristiques Petal', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Petal Length (cm)', fontsize=11)
        ax2.set_ylabel('Petal Width (cm)', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Classes réelles - Petal (généralement la meilleure séparation)
        ax3 = plt.subplot(1, 3, 3)
        colors_true = ['red', 'green', 'blue']
        for i, (color, name) in enumerate(zip(colors_true, self.target_names)):
            mask_true = self.data['true_class'] == i
            xy = self.X[mask_true, 2:]
            ax3.scatter(xy[:, 0], xy[:, 1], c=color, label=name,
                       edgecolors='k', s=100, alpha=0.7, linewidth=1.5)
        
        ax3.set_title('Classes Réelles - Petal', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Petal Length (cm)', fontsize=11)
        ax3.set_ylabel('Petal Width (cm)', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('iris_clustering_results.png', dpi=300, bbox_inches='tight')
        print("\n  ✓ Graphique sauvegardé: iris_clustering_results.png")
        plt.show()
        
        # Distribution des points par cluster
        fig, ax = plt.subplots(figsize=(12, 7))
        unique, counts = np.unique(self.labels, return_counts=True)
        labels_str = ['Bruit' if x == -1 else f'Cluster {x}' for x in unique]
        colors_bar = ['black' if x == -1 else plt.cm.Spectral(x/max(unique)) for x in unique]
        
        bars = ax.bar(labels_str, counts, color=colors_bar, edgecolor='black', 
                     linewidth=2, alpha=0.8)
        ax.set_xlabel('Clusters', fontsize=13, fontweight='bold')
        ax.set_ylabel('Nombre de points', fontsize=13, fontweight='bold')
        ax.set_title('Distribution des points par cluster (Dataset Iris)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs au-dessus des barres
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('iris_cluster_distribution.png', dpi=300, bbox_inches='tight')
        print("  ✓ Graphique sauvegardé: iris_cluster_distribution.png")
        plt.show()
        
        # Matrice de confusion entre clusters DBSCAN et classes réelles
        print("\n Comparaison DBSCAN vs Classes Réelles:")
        comparison = pd.crosstab(
            pd.Series(self.labels, name='Cluster DBSCAN'),
            pd.Series(self.data['species'], name='Espèce Réelle')
        )
        print(comparison)
    
    def interpret_results(self):
        """6. Interprétation des résultats"""
        print("\n" + "="*70)
        print("6. INTERPRÉTATION DES RÉSULTATS")
        print("="*70)
        
        print("\n Analyse détaillée des clusters:\n")
        
        # Analyser les caractéristiques de chaque cluster
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:
                continue
            
            mask = self.labels == cluster_id
            cluster_data = self.data[mask]
            
            print(f" Cluster {cluster_id}:")
            print(f"   Taille: {mask.sum()} échantillons ({mask.sum()/len(self.labels)*100:.1f}%)")
            
            # Statistiques moyennes
            means = cluster_data[self.feature_names].mean()
            print(f"   Caractéristiques moyennes:")
            for feat in self.feature_names:
                print(f"     • {feat}: {means[feat]:.2f} cm")
            
            # Distribution par espèce dans ce cluster
            species_dist = cluster_data['species'].value_counts()
            print(f"   Distribution par espèce:")
            for species, count in species_dist.items():
                print(f"     • {species}: {count} ({count/len(cluster_data)*100:.1f}%)")
            print()
        
        # Points de bruit
        noise_mask = self.labels == -1
        n_noise = noise_mask.sum()
        if n_noise > 0:
            print(f" Points de Bruit (Outliers):")
            print(f"   - Nombre: {n_noise}")
            print(f"   - Pourcentage: {n_noise/len(self.labels)*100:.1f}%")
            
            noise_data = self.data[noise_mask]
            species_noise = noise_data['species'].value_counts()
            print(f"   - Distribution par espèce:")
            for species, count in species_noise.items():
                print(f"     • {species}: {count}")
            print()
        
        # Observations sur la qualité du clustering
        print(" Observations:")
        
        # Calculer la pureté de chaque cluster
        for cluster_id in sorted(set(self.labels)):
            if cluster_id == -1:
                continue
            
            mask = self.labels == cluster_id
            cluster_data = self.data[mask]
            dominant_species = cluster_data['species'].mode()[0]
            purity = (cluster_data['species'] == dominant_species).sum() / len(cluster_data)
            
            print(f"   • Cluster {cluster_id}: Dominé par '{dominant_species}' (pureté: {purity:.1%})")
        
        print("\n Avantages de DBSCAN observés:")
        advantages = [
            "Détection automatique du nombre de clusters",
            "Identification des outliers (points de bruit)",
            "Clusters de formes arbitraires (non sphériques)",
            "Pas d'hypothèse sur la forme des clusters"
        ]
        for adv in advantages:
            print(f"   • {adv}")
        
        print("\n Limitations observées:")
        limitations = [
            "Sensible au choix des paramètres (eps, min_samples)",
            "Performance variable selon la densité des clusters",
            "Peut classer certains points de frontière comme bruit"
        ]
        for lim in limitations:
            print(f"   • {lim}")


def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print(" "*15 + "DBSCAN CLUSTERING - ANALYSE COMPLÈTE")
    print(" "*20 + "Dataset: Iris (sklearn)")
    print("="*70)
    
    # Créer l'instance d'analyse
    analysis = DBSCANAnalysis()
    
    # 1. Charger et préparer les données
    analysis.load_and_prepare_data()
    
    # 2. Prétraitement
    analysis.preprocess_data()
    
    # 3. Trouver epsilon optimal
    epsilon = analysis.find_optimal_epsilon()
    
    # 4. Appliquer DBSCAN
    analysis.apply_dbscan(eps=epsilon, min_samples=4)
    
    # 5. Évaluation et visualisation
    analysis.evaluate_and_visualize()
    
    # 6. Interprétation
    analysis.interpret_results()
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\n📁 Fichiers générés:")
    print("   • iris_k_distance_plot.png")
    print("   • iris_clustering_results.png")
    print("   • iris_cluster_distribution.png")
    print("\n💡 Le dataset Iris est différent du Wine dataset:")
    print("   - 150 échantillons (vs 178)")
    print("   - 4 caractéristiques (vs 13)")
    print("   - 3 espèces de fleurs (Setosa, Versicolor, Virginica)")
    print("   - Données morphologiques de fleurs (vs données chimiques de vin)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
