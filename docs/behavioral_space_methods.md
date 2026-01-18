# PRISM Behavioral Space Analysis Methods

A comprehensive catalog of mathematical methods for analyzing indicators in **behavioral space** - the geometric representation where each indicator is a point defined by its behavioral characteristics (Hurst exponent, entropy, volatility, etc.).

> **Key Distinction**: These methods operate on **static vectors** - positions in behavioral space at a given time. They do NOT require temporal sequences like signal topology methods (DTW, Granger, etc.).

---

## Currently Implemented

### Core Methods (In Use)

| Engine | What It Measures | PRISM Output |
|--------|------------------|--------------|
| **PCA** | Principal axes of variation in behavioral space | Dimensionality, variance explained, loadings |
| **Distance** | Euclidean, Mahalanobis, Cosine separation between indicators | Pairwise distance matrices |
| **Clustering** | K-means, hierarchical groupings of similar behaviors | Cluster assignments, silhouette scores |
| **Mutual Information** | Non-linear dependence between behavioral dimensions | Pairwise MI, dependency strength |

### Newly Implemented (High Priority)

| Engine | What It Measures | PRISM Output |
|--------|------------------|--------------|
| **MST (Minimum Spanning Tree)** | Skeleton of relationships connecting all indicators | Edge weights, hub nodes, diameter |
| **LOF (Local Outlier Factor)** | Density-based anomaly detection | Per-indicator LOF scores, outlier classification |
| **Convex Hull** | Geometric extent of indicator cloud | Volume, boundary indicators, centroid distances |

---

## Network / Graph Methods

Methods that treat indicators as nodes in a network, with edges defined by behavioral similarity.

### Minimum Spanning Tree ✅ (Implemented)
**Concept**: Connect all indicators with minimum total distance. Reveals the "backbone" of relationships.

**Metrics**:
- Total MST weight (sum of edge lengths)
- Average/max/min edge lengths
- Hub nodes (high-degree vertices)
- Leaf nodes (peripheral indicators)
- MST diameter (longest path)

**Interpretation**:
- Contracting MST → behavioral convergence (crisis?)
- High-degree hubs → "bridge" indicators connecting clusters
- Long edges → weakly connected indicator pairs

### k-Nearest Neighbors Graph
**Concept**: Connect each indicator to its k closest behavioral neighbors.

**Potential Metrics**:
- Average neighbor distance
- Clustering coefficient (local vs global)
- Connected components
- Average shortest path length

**Interpretation**:
- Dense clusters → behavioral regimes
- Isolated nodes → unusual behavioral signatures
- Small-world properties → efficient information flow

### Graph Laplacian / Spectral Analysis
**Concept**: Eigenvalues of the graph Laplacian reveal connectivity structure.

**Potential Metrics**:
- Algebraic connectivity (2nd smallest eigenvalue)
- Spectral gap
- Fiedler vector (community structure)
- Number of connected components

**Interpretation**:
- Large spectral gap → clear cluster separation
- Small algebraic connectivity → fragmented structure
- Fiedler vector signs → natural 2-way partition

---

## Manifold / Topology Methods

Methods that analyze the shape and structure of the indicator cloud in behavioral space.

### Local Outlier Factor ✅ (Implemented)
**Concept**: Compare local density around each indicator to its neighbors' density.

**Metrics**:
- LOF score per indicator (>1 = outlier)
- Outlier counts at thresholds (1.5, 2.0, 3.0)
- Most/least anomalous indicators

**Interpretation**:
- LOF > 2.0 → strongly unusual behavioral signature
- Sudden LOF increase → behavioral regime change
- Persistent outliers → structurally different indicators

### UMAP / t-SNE
**Concept**: Non-linear dimensionality reduction for visualization.

**Potential Metrics**:
- Cluster separation in 2D projection
- Trustworthiness (preservation of local structure)
- Continuity (preservation of global structure)

**Interpretation**:
- Tight clusters → distinct behavioral regimes
- Gradients → continuous behavioral transitions
- Isolated points → anomalous indicators

### Persistent Homology (TDA)
**Concept**: Track topological features (connected components, holes, voids) across scales.

**Potential Metrics**:
- Betti numbers (counts of features)
- Persistence diagrams
- Bottleneck distance between time periods

**Interpretation**:
- Persistent features → stable structural properties
- Short-lived features → noise
- Changing topology → structural regime shifts

### Isolation Forest
**Concept**: Tree-based anomaly detection. Anomalies are "easier to isolate."

**Potential Metrics**:
- Anomaly score per indicator
- Average path length
- Feature importance for isolation

**Interpretation**:
- Short path length → anomalous behavior
- Robust across different density patterns
- Complements density-based methods (LOF)

---

## Distribution / Density Methods

Methods that analyze the statistical distribution of indicators in behavioral space.

### Convex Hull ✅ (Implemented)
**Concept**: Smallest convex set containing all indicators.

**Metrics**:
- Hull volume (overall dispersion)
- Surface area
- Number of vertices (boundary indicators)
- Centroid and centroid distances
- Sphericity (shape regularity)

**Interpretation**:
- Contracting volume → convergence (risk?)
- Expanding volume → divergence
- Boundary indicators → extreme behavioral signatures
- Centroid distance → mainstream vs unusual

### Kernel Density Estimation
**Concept**: Estimate probability density in behavioral space.

**Potential Metrics**:
- Peak locations (behavioral modes)
- Number of modes (multimodality)
- Entropy of density distribution
- Per-indicator density values

**Interpretation**:
- Multiple peaks → distinct behavioral regimes
- High entropy → dispersed, no dominant pattern
- Low-density regions → unusual behavioral combinations

### Centroid Analysis
**Concept**: Track the center of mass of the indicator cloud.

**Potential Metrics**:
- Centroid coordinates (in behavioral dimensions)
- Centroid velocity (change over time)
- Distance from centroid per indicator
- Centroid stability (variance over windows)

**Interpretation**:
- Moving centroid → systematic behavioral shift
- Stable centroid → consistent regime
- Far from centroid → contrarian indicators

### Covariance Structure
**Concept**: Analyze the covariance/correlation between behavioral dimensions.

**Potential Metrics**:
- Condition number (numerical stability)
- Eigenvalue distribution
- Effective rank
- Determinant (generalized variance)

**Interpretation**:
- High condition number → near-singular, redundant dimensions
- Concentrated eigenvalues → strong principal axis
- Low determinant → compressed behavioral space

---

## Statistical / Machine Learning Methods

### Gaussian Mixture Models
**Concept**: Soft clustering with probabilistic assignments.

**Potential Metrics**:
- Optimal number of components (BIC/AIC)
- Per-indicator cluster probabilities
- Cluster means and covariances
- Overlap between clusters

**Interpretation**:
- Soft boundaries → gradual behavioral transitions
- High overlap → poorly separated regimes
- Component weights → regime prevalence

### Mahalanobis Distance Analysis
**Concept**: Distance accounting for correlation structure.

**Potential Metrics**:
- Mahalanobis distance from centroid
- Chi-squared significance test
- Outlier detection (distance > threshold)

**Interpretation**:
- Accounts for dimension correlations
- More appropriate than Euclidean for correlated behaviors
- Statistical significance of outlier status

### Procrustes Analysis
**Concept**: Compare shapes after optimal alignment (rotation, scaling, translation).

**Potential Metrics**:
- Procrustes distance between time periods
- Optimal rotation angle
- Scaling factor
- Residual after alignment

**Interpretation**:
- Small distance → similar behavioral structure across time
- Large rotation → behavioral space reorientation
- Compare regimes: 2008 vs 2020 "shape"

---

## Information-Theoretic Methods

### Joint Entropy
**Concept**: Total uncertainty in the joint distribution of behavioral dimensions.

**Potential Metrics**:
- Joint entropy H(X,Y,Z,...)
- Redundancy (mutual information overlap)
- Synergy (information only available jointly)

**Interpretation**:
- High joint entropy → diverse behaviors
- High redundancy → dimensions measure similar things
- Synergy → emergent behavioral patterns

### Information Geometry
**Concept**: Treat probability distributions as points on a manifold.

**Potential Metrics**:
- Fisher information matrix
- Geodesic distances between distributions
- Curvature of statistical manifold

**Interpretation**:
- Advanced framework for comparing behavioral regimes
- Natural metric for distribution comparison
- Connects to thermodynamic analogies

---

## Recommended Implementation Priority

### Phase 1: Core Behavioral Analysis ✅
- [x] PCA
- [x] Distance
- [x] Clustering
- [x] Mutual Information

### Phase 2: Structure and Anomalies ✅
- [x] Minimum Spanning Tree
- [x] Local Outlier Factor
- [x] Convex Hull

### Phase 3: Enhanced Network Analysis
- [ ] k-NN Graph with spectral analysis
- [ ] Graph Laplacian eigenvalues
- [ ] Community detection

### Phase 4: Distribution Analysis
- [ ] Gaussian Mixture Models
- [ ] Kernel Density Estimation
- [ ] Mahalanobis outlier detection

### Phase 5: Advanced / Research
- [ ] Persistent Homology (TDA)
- [ ] UMAP visualization
- [ ] Information geometry
- [ ] Procrustes temporal comparison

---

## Integration with PRISM Architecture

All behavioral space methods:
1. **Input**: Results from `results.vector` (behavioral descriptors)
2. **Operation**: Compare indicator positions in N-dimensional behavioral space
3. **Output**: Write to `results.geometry` (structural metrics)

**Data Flow**:
```
raw.observations → Vector Engines → results.vector
                                         ↓
                              Behavioral Space Matrix
                              (indicators × dimensions)
                                         ↓
                              Geometry Engines (this catalog)
                                         ↓
                              results.geometry + structure.*
```

**Key Distinction from Signal Topology Engines**:
| Behavioral Space Engines | Signal Topology Engines |
|--------------------------|---------------------|
| Compare static positions | Compare temporal evolution |
| Input: indicator × dimension matrix | Input: time × indicator matrix |
| Question: "How similar are behaviors?" | Question: "Who leads/lags whom?" |
| Works on single snapshot | Requires sequence of observations |

---

## References

- Mantegna, R.N. (1999). "Hierarchical structure" - MST method for correlation networks
- Breunig et al. (2000). "LOF: Identifying Density-Based Local Outliers" - LOF algorithm
- Barber et al. (1996). "The Quickhull Algorithm for Convex Hulls" - Convex hull computation
- Carlsson, G. (2009). "Topology and Data" - Persistent homology overview
- McInnes et al. (2018). "UMAP: Uniform Manifold Approximation" - UMAP method
