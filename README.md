# PSO Clustering Variations Explorer

This interactive Streamlit application demonstrates various Particle Swarm Optimization (PSO) variations for clustering tasks. The application allows users to experiment with different PSO algorithms, adjust parameters, and compare results.

## Features

- **Interactive UI**: Adjust algorithm parameters and see results in real-time
- **Multiple PSO Variants**: Compare 6 different PSO variations and K-Means
- **Auto-Parameter Tuning**: Automatically find the best parameters for each algorithm
- **Visualization**: 2D, 3D, and PCA visualizations of clustering results
- **Comparative Analysis**: Compare algorithms using inertia and silhouette scores
- **Customizable Parameters**: Tune each algorithm to see how parameters affect performance

## PSO Variations Included

1. **K-Means** (Baseline for comparison)
2. **Global-Best PSO** - Standard PSO variant
3. **Local-Best PSO** - Uses ring neighborhood topology
4. **Linear Inertia PSO** - Linearly decreasing inertia weight
5. **Constriction Factor PSO** - Uses Clerc's constriction coefficient
6. **Velocity-Clamped PSO** - Limits maximum velocity
7. **PSO with Gaussian Mutation** - Adds random perturbations to particles
8. **Auto-Parameter Tuning** - Finds optimal parameters for all algorithms

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the batch file

Simply double-click on `run_streamlit_app.bat` to start the application.

### Option 2: Run from command line

```
streamlit run app/streamlit_app.py
```

## Dataset

The application comes with the Mall Customer dataset with features:

- Age
- Annual Income (k$)
- Spending Score (1-100)

You can also upload your own CSV dataset.

## Algorithm Descriptions

### K-Means (Baseline)

K-Means works by initializing k centroids randomly, then iteratively assigning points to the nearest centroid and updating centroids based on the mean of assigned points.

### Global-Best PSO

Each particle represents a set of centroids. Particles update their velocities based on their personal best position and the global best position found by any particle.

### Local-Best PSO

Similar to Global-Best PSO, but particles are influenced by their neighborhood best rather than the global best, which helps maintain diversity.

### Linear Inertia PSO

Uses a linearly decreasing inertia weight to balance exploration and exploitation throughout the search process.

### Constriction Factor PSO

Uses a constriction coefficient to control particle velocity, providing theoretical convergence guarantees.

### Velocity-Clamped PSO

Limits the maximum velocity of particles to prevent excessive steps and improve convergence.

### PSO with Gaussian Mutation

Introduces random perturbations to some particles during the search to enhance exploration and avoid premature convergence.

### Auto-Parameter Tuning

Automatically runs multiple iterations of each algorithm with different parameter sets to find the optimal configuration. This feature helps identify the best parameters without manual trial and error.

## How to Interpret Results

- **Inertia**: Sum of squared distances from points to their assigned centroids. Lower is better.
- **Silhouette Score**: Measure of how similar points are to their own cluster compared to other clusters. Range from -1 to 1, with higher values indicating better clustering.

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. In Proceedings of ICNN'95.
2. Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space.
3. Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
