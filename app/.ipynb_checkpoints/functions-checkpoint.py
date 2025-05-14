import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
import math
# Set global random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def pso_global(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5):
    """Global-best PSO for clustering: returns flattened best centroids and best fitness."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Feature-wise bounds for initialization
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize swarm positions and velocities
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))

    # Personal bests
    pbest_pos = positions.copy()
    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)
    pbest_fit = np.array([fitness(p) for p in positions])

    # Global best
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # PSO main loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()

    return gbest_pos, gbest_fit

def pso_local(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5):
    """Local-best PSO for clustering with ring neighborhood: returns best centroids and fitness."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Feature-wise bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize positions & velocities
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))

    # Personal bests
    pbest_pos = positions.copy()
    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)
    pbest_fit = np.array([fitness(p) for p in positions])

    # Local best positions (initially same as personal)
    lbest_pos = pbest_pos.copy()
    lbest_fit = pbest_fit.copy()

    # Update local best for each particle's ring neighborhood
    def update_local_bests():
        for i in range(swarm_size):
            neighbors = [(i-1) % swarm_size, i, (i+1) % swarm_size]
            best_idx = min(neighbors, key=lambda idx: pbest_fit[idx])
            lbest_pos[i] = pbest_pos[best_idx]
            lbest_fit[i] = pbest_fit[best_idx]

    update_local_bests()

    # PSO main loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (lbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]
        update_local_bests()

    # Choose best among all personal bests
    best_idx = np.argmin(pbest_fit)
    return pbest_pos[best_idx], pbest_fit[best_idx]

def pso_linear_inertia(X, k, swarm_size=30, max_iters=100, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):
    """PSO with linearly decreasing inertia weight."""
    n_samples, n_features = X.shape
    dim = k * n_features
    
    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)
    
    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()
    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)
    pbest_fit = np.array([fitness(p) for p in positions])
    
    # Global best init
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    # Main loop
    for iter in range(max_iters):
        # Update inertia
        w = w_max - (w_max - w_min) * (iter / (max_iters - 1))
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)
        
        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]
        
        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
    
    return gbest_pos, gbest_fit

def pso_constriction(X, k, swarm_size=30, max_iters=100, c1=2.05, c2=2.05):
    """PSO with constriction factor (Clerc's constriction coefficient)."""
    n_samples, n_features = X.shape
    dim = k * n_features
    
    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)
    
    # Calculate constriction coefficient
    phi = c1 + c2
    chi = 2 / abs(2 - phi - math.sqrt(phi**2 - 4*phi))
    
    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()
    
    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)
    
    pbest_fit = np.array([fitness(p) for p in positions])
    # Global best
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]
    
    # PSO loop
    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = chi * (velocities
                            + c1 * r1 * (pbest_pos - positions)
                            + c2 * r2 * (gbest_pos - positions))
        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)
        
        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]
        
        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()
    
    return gbest_pos, gbest_fit

def pso_velocity_clamped(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5, vmax_frac=0.2):
    """PSO with velocity clamping."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Max velocity per dimension
    vmax = (pos_max - pos_min) * vmax_frac

    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.random.uniform(-vmax, vmax, (swarm_size, dim))
    pbest_pos = positions.copy()

    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)

    pbest_fit = np.array([fitness(p) for p in positions])
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))

        # Clamp velocities
        velocities = np.clip(velocities, -vmax, vmax)

        positions += velocities
        positions = np.clip(positions, pos_min, pos_max)

        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()

    return gbest_pos, gbest_fit

def pso_mutation(X, k, swarm_size=30, max_iters=100, w=0.5, c1=1.5, c2=1.5, mutation_prob=0.2, mutation_scale=0.05):
    """PSO with Gaussian mutation injected into particles."""
    n_samples, n_features = X.shape
    dim = k * n_features

    # Bounds
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    pos_min = np.tile(data_min, k)
    pos_max = np.tile(data_max, k)

    # Initialize
    positions = np.random.uniform(pos_min, pos_max, (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))
    pbest_pos = positions.copy()

    def fitness(pos):
        centers = pos.reshape(k, n_features)
        _, dists = pairwise_distances_argmin_min(X, centers)
        return np.sum(dists**2)

    pbest_fit = np.array([fitness(p) for p in positions])
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(max_iters):
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions += velocities

        # Inject Gaussian mutation into a subset of particles
        for i in range(swarm_size):
            if np.random.rand() < mutation_prob:
                mutation = np.random.normal(0, mutation_scale, dim)
                positions[i] += mutation

        positions = np.clip(positions, pos_min, pos_max)

        fitness_vals = np.array([fitness(p) for p in positions])
        improved = fitness_vals < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness_vals[improved]

        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()

    return gbest_pos, gbest_fit