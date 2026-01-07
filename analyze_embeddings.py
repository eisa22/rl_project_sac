"""
Analyze and Visualize Task Embeddings

This script loads a trained model and visualizes the learned task embeddings
to understand which tasks the agent considers similar.

Usage:
    python analyze_embeddings.py --model_path ./models_mt10_embeddings/mt10_embeddings/final_model.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns


def load_embeddings(model_path):
    """Load task embeddings from trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    actor_emb = checkpoint['actor_task_embeddings'].numpy()
    q1_emb = checkpoint['q1_task_embeddings'].numpy()
    q2_emb = checkpoint['q2_task_embeddings'].numpy()
    
    return actor_emb, q1_emb, q2_emb


def visualize_embeddings_2d(embeddings, task_names, method='tsne', title='Task Embeddings'):
    """
    Visualize embeddings in 2D using t-SNE or PCA.
    
    Args:
        embeddings: [num_tasks, embedding_dim] numpy array
        task_names: list of task names
        method: 'tsne' or 'pca'
        title: plot title
    """
    if method == 'tsne':
        # t-SNE: Captures non-linear relationships
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(task_names)-1))
    else:
        # PCA: Linear projection
        reducer = PCA(n_components=2)
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6)
    
    for i, name in enumerate(task_names):
        plt.annotate(name, 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=10,
                    ha='center',
                    va='bottom')
    
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_distance_matrix(embeddings, task_names, title='Task Distance Matrix'):
    """
    Plot pairwise distances between task embeddings.
    Shows which tasks are similar (small distance) vs different (large distance).
    """
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(embeddings, embeddings, metric='euclidean')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, 
                xticklabels=task_names,
                yticklabels=task_names,
                annot=True,
                fmt='.2f',
                cmap='viridis',
                cbar_kws={'label': 'Euclidean Distance'})
    plt.title(title)
    plt.tight_layout()
    return plt


def analyze_embedding_dimensions(embeddings, task_names):
    """
    Analyze which embedding dimensions are most important.
    """
    embedding_dim = embeddings.shape[1]
    
    # Compute variance per dimension
    variances = np.var(embeddings, axis=0)
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(embedding_dim), variances)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Variance')
    plt.title('Variance per Embedding Dimension (higher = more informative)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"\n📊 Embedding Dimension Analysis:")
    print(f"  Total Dimensions: {embedding_dim}")
    print(f"  Mean Variance: {np.mean(variances):.4f}")
    print(f"  Max Variance: {np.max(variances):.4f} (dimension {np.argmax(variances)})")
    print(f"  Min Variance: {np.min(variances):.4f} (dimension {np.argmin(variances)})")
    
    # Find most varying dimensions
    top_dims = np.argsort(variances)[-3:][::-1]
    print(f"\n  Top 3 Most Informative Dimensions: {top_dims.tolist()}")
    
    return plt


def main():
    parser = argparse.ArgumentParser(description='Analyze task embeddings')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model .pt file')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to disk instead of showing')
    args = parser.parse_args()
    
    # MT10 task names (in order)
    task_names = [
        'reach-v2', 'push-v2', 'pick-place-v2', 
        'door-open-v2', 'drawer-open-v2', 'drawer-close-v2',
        'button-press-topdown-v2', 'peg-insert-side-v2',
        'window-open-v2', 'window-close-v2'
    ]
    
    print(f"Loading embeddings from: {args.model_path}\n")
    
    # Load embeddings
    actor_emb, q1_emb, q2_emb = load_embeddings(args.model_path)
    
    print(f"✓ Loaded embeddings:")
    print(f"  Actor: {actor_emb.shape}")
    print(f"  Critic Q1: {q1_emb.shape}")
    print(f"  Critic Q2: {q2_emb.shape}")
    
    # Analyze actor embeddings
    print("\n" + "="*70)
    print("ACTOR EMBEDDINGS ANALYSIS")
    print("="*70)
    
    # 1. t-SNE visualization
    print("\n1. Creating t-SNE visualization...")
    plt_tsne = visualize_embeddings_2d(actor_emb, task_names, method='tsne', 
                                       title='Actor Task Embeddings (t-SNE)')
    if args.save_plots:
        plt_tsne.savefig('embeddings_tsne.png', dpi=150)
        print("   Saved: embeddings_tsne.png")
    else:
        plt_tsne.show()
    
    # 2. PCA visualization
    print("\n2. Creating PCA visualization...")
    plt_pca = visualize_embeddings_2d(actor_emb, task_names, method='pca',
                                      title='Actor Task Embeddings (PCA)')
    if args.save_plots:
        plt_pca.savefig('embeddings_pca.png', dpi=150)
        print("   Saved: embeddings_pca.png")
    else:
        plt_pca.show()
    
    # 3. Distance matrix
    print("\n3. Creating distance matrix...")
    plt_dist = plot_distance_matrix(actor_emb, task_names, 
                                    title='Actor Task Embedding Distances')
    if args.save_plots:
        plt_dist.savefig('embeddings_distances.png', dpi=150)
        print("   Saved: embeddings_distances.png")
    else:
        plt_dist.show()
    
    # 4. Dimension analysis
    print("\n4. Analyzing embedding dimensions...")
    plt_dims = analyze_embedding_dimensions(actor_emb, task_names)
    if args.save_plots:
        plt_dims.savefig('embeddings_dimensions.png', dpi=150)
        print("   Saved: embeddings_dimensions.png")
    else:
        plt_dims.show()
    
    # Find most similar tasks
    from scipy.spatial.distance import cdist
    distances = cdist(actor_emb, actor_emb, metric='euclidean')
    
    print("\n" + "="*70)
    print("TASK SIMILARITY RANKING")
    print("="*70)
    
    for i, task in enumerate(task_names):
        # Get distances to other tasks (excluding self)
        dists = [(task_names[j], distances[i, j]) for j in range(len(task_names)) if i != j]
        dists.sort(key=lambda x: x[1])
        
        print(f"\n{task}:")
        print(f"  Most similar:   {dists[0][0]:25s} (distance: {dists[0][1]:.3f})")
        print(f"  Least similar:  {dists[-1][0]:25s} (distance: {dists[-1][1]:.3f})")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Tasks that are CLOSE in embedding space:
  → Agent considers them similar (shared strategies)
  → Likely benefit from transfer learning
  → Example: reach-v2 and push-v2 (both involve arm movement)

Tasks that are FAR in embedding space:
  → Agent considers them different (distinct strategies)
  → Less transfer between them
  → Example: reach-v2 and door-open-v2 (different mechanics)

High variance dimensions:
  → Important for distinguishing tasks
  → Agent learned to use these features

Low variance dimensions:
  → Less useful for task differentiation
  → Could potentially reduce embedding_dim
    """)


if __name__ == '__main__':
    main()
