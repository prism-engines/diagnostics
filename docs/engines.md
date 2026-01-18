# PRISM Engine Reference

> Mathematical measurement engines for signal topology analysis.

---

## Three Engine Classes

PRISM engines divide into three classes based on what they measure and how:

| Class | Scope | Time Required? | Question Answered |
|-------|-------|----------------|-------------------|
| **Vector** | One indicator | No | "What are the intrinsic properties of this series?" |
| **Geometry** | Multiple indicators | No | "What is the static relational structure?" |
| **State** | Multiple indicators | Yes | "How do relationships evolve through time?" |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRISM Engine Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   VECTOR ENGINES              GEOMETRY ENGINES           STATE ENGINES   │
│   (Intrinsic)                 (Relational)               (Temporal)      │
│   ─────────────               ────────────               ─────────────   │
│                                                                          │
│   Series A ──► [Hurst]        Behavioral     ──► [PCA]   Signal Topology    │
│   Series B ──► [Entropy]       Vectors       ──► [MST]    + History     │
│   Series C ──► [GARCH]          ↓            ──► [LOF]       ↓          │
│        ↓                    Static Structure            ──► [Granger]   │
│   Behavioral                 (no time arrow)            ──► [DTW]       │
│   Descriptors                                           ──► [Cointegr]  │
│                                                                          │
│   "What is each             "How do they               "How do they     │
│    series doing?"            relate now?"               evolve together?"│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Vector Engines

**Purpose:** Measure intrinsic properties of individual signal topology.

Each vector engine answers: *"What is this series doing on its own?"*

These measurements are **intrinsic** — they don't depend on other indicators and produce a behavioral descriptor for each series.

| Engine | Measures | Output Dimensions |
|--------|----------|-------------------|
| Hurst | Memory/persistence | 1 |
| Entropy | Complexity/predictability | 2-3 |
| GARCH | Volatility clustering | 3-4 |
| Wavelet | Multi-scale energy | 4-6 |
| Spectral | Frequency content | 3-4 |
| Lyapunov | Chaos/sensitivity | 1-2 |
| RQA | Recurrence patterns | 4-6 |

---

### Hurst Exponent

**Measures:** Long-term memory and persistence.

**Question:** Does this series trend, mean-revert, or walk randomly?

**Why it matters:** A trending series (H > 0.5) behaves fundamentally differently from a mean-reverting one (H < 0.5). This determines whether momentum or reversion strategies apply, and reveals the fractal nature of the signal topology.

#### Formula: Rescaled Range Method

$$H = \frac{\log(R/S)}{\log(n)}$$

where $R/S$ is the rescaled range:

$$\frac{R}{S} = \frac{\max_k Y_k - \min_k Y_k}{\sigma}$$

and $Y_k = \sum_{i=1}^k (x_i - \bar{x})$ is cumulative deviation from mean.

**Interpretation:**

| Value | Meaning | Behavior |
|-------|---------|----------|
| H = 0.5 | No memory | Random walk (Brownian motion) |
| H > 0.5 | Persistent | Trends continue (momentum) |
| H < 0.5 | Anti-persistent | Mean-reverting |

**Minimum samples:** 100

---

### Entropy

**Measures:** Complexity and unpredictability.

**Question:** How disordered or structured is this series?

**Why it matters:** Low entropy means predictable patterns exist; high entropy means noise-like behavior. This determines whether pattern-based analysis is feasible and quantifies the information content of the series.

#### Formula: Permutation Entropy

$$H_p = -\sum_{\pi} p(\pi) \log_2 p(\pi)$$

Counts ordinal patterns (up-down sequences) of embedding dimension $d$. Normalized to [0,1]:

$$H_p^{norm} = \frac{H_p}{\log_2(d!)}$$

#### Formula: Sample Entropy

$$\text{SampEn}(m, r, N) = -\ln\frac{A}{B}$$

where:
- $A$ = number of template matches at length $m+1$
- $B$ = number of template matches at length $m$
- $r$ = tolerance threshold (typically 0.2 × std)

**Interpretation:**

| Value | Meaning |
|-------|---------|
| Low (~0) | Regular, predictable patterns |
| High (~1) | Complex, random-like |

**Minimum samples:** 50

---

### GARCH Volatility

**Measures:** Volatility clustering and conditional variance dynamics.

**Question:** Does uncertainty cluster in time? Are calm and turbulent periods distinct?

**Why it matters:** GARCH captures the empirical fact that large moves cluster together (volatility clustering). This is essential for risk measurement and reveals the heteroskedastic nature of many signal topology.

#### Formula: GARCH(1,1) Model

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where:
- $\omega$ = long-run variance constant
- $\alpha$ = shock impact (ARCH term) — how much yesterday's surprise affects today's variance
- $\beta$ = persistence (GARCH term) — how long elevated volatility lasts
- $\alpha + \beta$ = total persistence (must be < 1 for stationarity)

#### Unconditional Variance

$$\sigma^2 = \frac{\omega}{1 - \alpha - \beta}$$

**Interpretation:**

| $\alpha + \beta$ | Meaning |
|------------------|---------|
| Near 0 | Volatility reverts quickly to mean |
| Near 1 | Shocks persist for extended periods |
| = 1 | Integrated GARCH (infinite persistence) |

**Minimum samples:** 63

---

### Wavelet Analysis

**Measures:** Energy distribution across time scales.

**Question:** What scales (fast vs slow oscillations) dominate this series?

**Why it matters:** A series may have high-frequency noise but low-frequency trends. Wavelets decompose the signal into scale-specific components, revealing structure invisible in the time domain. Unlike Fourier, wavelets localize in both time and frequency.

#### Formula: Continuous Wavelet Transform

$$W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt$$

where:
- $a$ = scale parameter (larger = slower oscillations)
- $b$ = translation (time position)
- $\psi$ = mother wavelet (e.g., Morlet, Daubechies)

#### Scale Energy

$$E(a) = \sum_b |W(a, b)|^2$$

#### Wavelet Entropy

$$S_W = -\sum_j p_j \ln p_j, \quad p_j = \frac{E_j}{\sum_k E_k}$$

**Outputs:** Energy in high/mid/low frequency bands, dominant scale, wavelet entropy.

**Minimum samples:** 64

---

### Spectral Analysis

**Measures:** Frequency content via Fourier decomposition.

**Question:** What periodicities exist in this series?

**Why it matters:** Many phenomena are cyclical (daily, weekly, seasonal, business cycle). Spectral analysis reveals dominant frequencies invisible in the time domain and quantifies the "color" of noise (white, pink, brown).

#### Formula: Power Spectral Density

$$P(f) = |X(f)|^2 = \left|\int_{-\infty}^{\infty} x(t) e^{-2\pi i f t} dt\right|^2$$

#### Spectral Flatness (Wiener Entropy)

$$\text{flatness} = \frac{\exp\left(\frac{1}{N}\sum_f \ln P(f)\right)}{\frac{1}{N}\sum_f P(f)} = \frac{\text{geometric mean}}{\text{arithmetic mean}}$$

#### Spectral Centroid

$$f_c = \frac{\sum_f f \cdot P(f)}{\sum_f P(f)}$$

**Interpretation:**

| Flatness | Meaning |
|----------|---------|
| Near 0 | Tonal (dominant frequencies) |
| Near 1 | Noise-like (flat spectrum) |

**Minimum samples:** 64

---

### Lyapunov Exponent

**Measures:** Sensitivity to initial conditions (chaos indicator).

**Question:** Do nearby trajectories in phase space diverge exponentially?

**Why it matters:** Positive Lyapunov exponent indicates chaos — deterministic but fundamentally unpredictable beyond a horizon. This distinguishes chaotic dynamics from noise and reveals the limits of predictability.

#### Formula: Largest Lyapunov Exponent

$$\lambda_1 = \lim_{t \to \infty} \frac{1}{t} \ln \frac{\|\delta \mathbf{x}(t)\|}{\|\delta \mathbf{x}(0)\|}$$

Rate of exponential divergence between initially nearby trajectories in reconstructed phase space.

#### Practical Estimation (Rosenstein method)

$$\lambda_1 \approx \frac{1}{\Delta t} \langle \ln d_j(i) \rangle$$

where $d_j(i)$ is the distance between trajectory $j$ and its nearest neighbor after $i$ time steps.

**Interpretation:**

| Value | Meaning |
|-------|---------|
| λ > 0 | Chaotic (exponential divergence) |
| λ < 0 | Stable attractor (convergence) |
| λ ≈ 0 | Edge of chaos / periodic |

**Minimum samples:** 200

---

### Recurrence Quantification Analysis (RQA)

**Measures:** Recurrence patterns in reconstructed phase space.

**Question:** How often does the system revisit similar states?

**Why it matters:** RQA detects deterministic structure that linear methods miss. High determinism means the system follows dynamical rules; high laminarity indicates intermittency and regime transitions. Captures nonlinear dynamics without assuming a model.

#### Formula: Recurrence Matrix

$$R_{i,j} = \Theta(\epsilon - \|\mathbf{x}_i - \mathbf{x}_j\|)$$

where $\Theta$ is the Heaviside step function. Entry is 1 if states $i$ and $j$ are within threshold $\epsilon$.

#### Recurrence Rate

$$RR = \frac{1}{N^2} \sum_{i,j} R_{i,j}$$

#### Determinism

$$DET = \frac{\sum_{l=l_{min}}^{N} l \cdot P(l)}{\sum_{i,j} R_{i,j}}$$

where $P(l)$ = histogram of diagonal line lengths. High DET indicates deterministic dynamics.

#### Laminarity

$$LAM = \frac{\sum_{v=v_{min}}^{N} v \cdot P(v)}{\sum_{i,j} R_{i,j}}$$

where $P(v)$ = histogram of vertical line lengths. High LAM indicates intermittency.

**Minimum samples:** 100

---

## Geometry Engines

**Purpose:** Measure static relational structure between indicators.

Each geometry engine answers: *"What is the structural relationship right now?"*

These measurements are **relational** — they require multiple indicators but do not need time ordering. They analyze positions in behavioral space, not trajectories.

| Engine | Measures | Input |
|--------|----------|-------|
| PCA | Shared variance structure | Behavioral vectors |
| Clustering | Natural groupings | Behavioral vectors |
| Distance | Pairwise dissimilarity | Behavioral vectors |
| Mutual Information | Nonlinear dependence | Behavioral vectors |
| Copula | Tail dependence | Behavioral vectors |
| MST | Minimum spanning tree | Distance matrix |
| LOF | Local outlier factor | Behavioral vectors |
| Convex Hull | Geometric extent | Behavioral vectors |

---

### Principal Component Analysis (PCA)

**Measures:** Shared variance structure and dimensionality.

**Question:** How much movement is common vs idiosyncratic?

**Why it matters:** If PC1 explains 80% of variance, indicators move together as a bloc. If variance is spread across many components, they're mostly independent. PCA reveals the true dimensionality and factor structure of the system.

#### Formula: Eigendecomposition

$$\Sigma \mathbf{v}_k = \lambda_k \mathbf{v}_k$$

where:
- $\Sigma$ = covariance matrix
- $\lambda_k$ = variance explained by component $k$
- $\mathbf{v}_k$ = loadings (how each indicator contributes)

#### Variance Explained

$$VE_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

#### Effective Rank (Intrinsic Dimensionality)

$$r_{eff} = \exp\left(-\sum_k \hat{\lambda}_k \ln \hat{\lambda}_k\right)$$

where $\hat{\lambda}_k = \lambda_k / \sum_i \lambda_i$. Ranges from 1 (everything moves together) to N (all independent).

**Minimum indicators:** 3

---

### Clustering

**Measures:** Natural groupings among indicators.

**Question:** Which indicators behave similarly?

**Why it matters:** Indicators may cluster by domain, regime, or latent factor. Clustering reveals structure that pairwise correlation misses and identifies behavioral regimes.

#### Formula: Silhouette Score

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:
- $a(i)$ = mean distance to same-cluster members (cohesion)
- $b(i)$ = mean distance to nearest other cluster (separation)

#### Calinski-Harabasz Index

$$CH = \frac{SS_B / (k-1)}{SS_W / (n-k)}$$

Between-cluster to within-cluster variance ratio.

**Interpretation:**

| Silhouette | Meaning |
|------------|---------|
| s ≈ 1 | Well-separated, dense clusters |
| s ≈ 0 | Overlapping clusters |
| s < 0 | Likely misclassified |

**Minimum indicators:** 4

---

### Distance Metrics

**Measures:** Pairwise dissimilarity between indicators.

**Question:** How far apart are these indicators in behavioral space?

**Why it matters:** Distance is the foundation for clustering, MST, and anomaly detection. Different metrics capture different notions of similarity.

#### Euclidean Distance

$$d_{ij} = \sqrt{\sum_k (x_{ik} - x_{jk})^2}$$

Standard geometric distance.

#### Mahalanobis Distance

$$d_i = \sqrt{(\mathbf{x}_i - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}_i - \boldsymbol{\mu})}$$

Accounts for correlation structure; measures distance in units of standard deviation.

#### Cosine Distance

$$d_{ij} = 1 - \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}$$

Measures angular separation; invariant to magnitude.

#### Correlation Distance

$$d_{ij} = 1 - \rho_{ij}$$

Converts correlation to distance.

---

### Mutual Information

**Measures:** Total statistical dependence (linear + nonlinear).

**Question:** How much does knowing X tell you about Y?

**Why it matters:** Correlation only captures linear relationships. Mutual information captures any dependence — including nonlinear, threshold, and complex relationships invisible to Pearson correlation.

#### Formula

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

$$= \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Reduction in uncertainty about Y when X is known.

#### Normalized Mutual Information

$$NMI = \frac{2 \cdot I(X;Y)}{H(X) + H(Y)}$$

Ranges from 0 (independent) to 1 (deterministic relationship).

**Minimum samples:** 50

---

### Copula Analysis

**Measures:** Dependence structure, especially in tails.

**Question:** Do extreme events happen together?

**Why it matters:** Normal correlation fails in tails — the most critical region for risk. Copulas reveal whether crashes or rallies are synchronized, independent of marginal distributions. Critical for understanding systemic risk.

#### Tail Dependence Coefficients

$$\lambda_U = \lim_{u \to 1} P(V > u \mid U > u)$$

$$\lambda_L = \lim_{u \to 0} P(V < u \mid U < u)$$

Probability of joint extreme given one extreme.

#### Kendall's Tau

$$\tau = \frac{\text{concordant pairs} - \text{discordant pairs}}{\binom{n}{2}}$$

Rank-based correlation, robust to outliers and nonlinearity.

#### Spearman's Rho

$$\rho_S = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}$$

Correlation of ranks.

**Minimum samples:** 100

---

### Minimum Spanning Tree (MST)

**Measures:** Essential connectivity structure.

**Question:** What is the minimal network connecting all indicators?

**Why it matters:** MST extracts the backbone of relationships, filtering noise. Central nodes in the MST are systemically important; peripheral nodes are idiosyncratic.

#### Construction (Prim/Kruskal)

Given distance matrix $D$, find tree $T$ minimizing:

$$\sum_{(i,j) \in T} d_{ij}$$

subject to $T$ connecting all nodes with $n-1$ edges.

#### Centrality Measures

- **Degree centrality:** Number of edges
- **Betweenness:** Fraction of shortest paths through node
- **Closeness:** Inverse of average distance to all nodes

---

### Local Outlier Factor (LOF)

**Measures:** Anomaly score based on local density.

**Question:** Which indicators are behavioral outliers?

**Why it matters:** Outliers may signal regime changes, data errors, or unique dynamics. LOF detects outliers relative to local neighborhood density, robust to varying densities across the space.

#### Formula

$$LOF(x) = \frac{\sum_{y \in N_k(x)} \frac{lrd(y)}{lrd(x)}}{|N_k(x)|}$$

where local reachability density:

$$lrd(x) = \frac{|N_k(x)|}{\sum_{y \in N_k(x)} \text{reach-dist}_k(x, y)}$$

**Interpretation:**

| LOF | Meaning |
|-----|---------|
| ≈ 1 | Normal (similar density to neighbors) |
| > 1 | Outlier (lower density than neighbors) |
| < 1 | Dense region (higher density than neighbors) |

---

### Convex Hull

**Measures:** Geometric extent and centrality.

**Question:** What is the boundary of the behavioral space?

**Why it matters:** Indicators on the convex hull are extreme in some dimension. Distance to hull center measures how "typical" an indicator is. Hull volume measures the spread of the system.

#### Centrality Score

$$c_i = 1 - \frac{\|\mathbf{x}_i - \mathbf{c}\|}{\max_j \|\mathbf{x}_j - \mathbf{c}\|}$$

where $\mathbf{c}$ is the centroid. Score of 1 = at center; 0 = at boundary.

---

## State Engines (Temporal)

**Purpose:** Measure how relationships evolve through time.

Each state engine answers: *"How does the system dynamics unfold?"*

These measurements are **temporal** — they require time ordering and analyze trajectories, causality, and dynamic coupling. They answer questions about information flow and temporal structure that static geometry cannot.

| Engine | Measures | Time Structure |
|--------|----------|----------------|
| Granger Causality | Predictive information flow | Lagged regression |
| Cross-Correlation | Lead/lag synchronization | Lagged correlation |
| Cointegration | Long-run equilibrium | Error correction |
| DTW | Shape similarity | Warping alignment |
| DMD | Coherent modes | Spectral decomposition |
| Transfer Entropy | Directed information flow | Conditional entropy |

---

### Granger Causality

**Measures:** Predictive information flow between series.

**Question:** Does knowing X's past help predict Y's future?

**Why it matters:** Granger causality tests whether one series contains predictive information about another beyond its own history. This reveals directional influence and temporal precedence in the system.

#### Formula: F-Test

Compare restricted model (Y's past only):
$$Y_t = \alpha + \sum_{i=1}^p \beta_i Y_{t-i} + \epsilon_t$$

To unrestricted model (Y's and X's past):
$$Y_t = \alpha + \sum_{i=1}^p \beta_i Y_{t-i} + \sum_{j=1}^p \gamma_j X_{t-j} + \epsilon_t$$

$$F = \frac{(SSR_R - SSR_U) / p}{SSR_U / (n - 2p - 1)}$$

Significant F (p < 0.05) means X Granger-causes Y.

**Note:** Both series must be stationary. Difference if needed.

**Minimum samples:** 50

---

### Cross-Correlation

**Measures:** Lead/lag synchronization.

**Question:** Does one series lead or lag another? By how much?

**Why it matters:** In dynamic systems, some variables respond before others. Cross-correlation reveals temporal ordering and the timescale of coupling.

#### Formula

$$\rho_{xy}(k) = \frac{\text{Cov}(x_t, y_{t+k})}{\sigma_x \sigma_y} = \frac{E[(x_t - \mu_x)(y_{t+k} - \mu_y)]}{\sigma_x \sigma_y}$$

**Interpretation:**

| Peak at lag k | Meaning |
|---------------|---------|
| k > 0 | X leads Y by k periods |
| k < 0 | Y leads X by |k| periods |
| k = 0 | Synchronous movement |

**Minimum samples:** 50

---

### Cointegration

**Measures:** Long-run equilibrium relationship.

**Question:** Do these non-stationary series share a common stochastic trend?

**Why it matters:** Two random walks may drift apart forever, or stay bound by an equilibrium. Cointegration finds pairs that mean-revert relative to each other — indicating shared underlying dynamics.

#### Formula: Engle-Granger Method

1. Regress: $Y_t = \alpha + \beta X_t + \epsilon_t$
2. Test residuals $\epsilon_t$ for stationarity using ADF test

If residuals are stationary (reject unit root), the series are cointegrated with cointegrating vector $(1, -\beta)$.

#### Spread Half-Life

$$\tau_{1/2} = -\frac{\ln 2}{\ln \phi}$$

where $\phi$ is the AR(1) coefficient of the spread. Measures how fast the spread mean-reverts (in periods).

#### Error Correction Model

$$\Delta Y_t = \alpha(\beta' \mathbf{X}_{t-1}) + \Gamma \Delta \mathbf{X}_{t-1} + \epsilon_t$$

Speed of adjustment $\alpha$ indicates how quickly deviations correct.

**Minimum samples:** 100

---

### Dynamic Time Warping (DTW)

**Measures:** Shape similarity with temporal flexibility.

**Question:** Do these series have similar shapes, ignoring exact timing?

**Why it matters:** Two series may have identical patterns but stretched, compressed, or shifted in time. DTW finds optimal nonlinear alignment, revealing similarity that correlation at fixed lag would miss.

#### Formula: Dynamic Programming

$$D(i,j) = d(x_i, y_j) + \min\begin{cases} D(i-1,j) \\ D(i,j-1) \\ D(i-1,j-1) \end{cases}$$

where $d(x_i, y_j) = |x_i - y_j|^2$ is the local distance.

#### DTW Distance

$$DTW(X,Y) = D(n,m)$$

The accumulated cost of the optimal warping path.

#### DTW Similarity

$$\text{similarity} = \frac{1}{1 + DTW(X,Y)}$$

**Minimum samples:** 30

---

### Dynamic Mode Decomposition (DMD)

**Measures:** Coherent spatiotemporal modes and frequencies.

**Question:** What are the dominant dynamic patterns across indicators?

**Why it matters:** DMD extracts oscillation frequencies and growth/decay rates from multivariate signal topology. It approximates the underlying linear operator governing system dynamics, revealing modes that may not be apparent from individual series analysis.

#### Formula

Given data matrices $\mathbf{X} = [\mathbf{x}_1, ..., \mathbf{x}_{m-1}]$ and $\mathbf{X}' = [\mathbf{x}_2, ..., \mathbf{x}_m]$:

$$\mathbf{X}' \approx \mathbf{A}\mathbf{X}$$

DMD finds eigenvalues $\lambda_j$ and modes $\phi_j$ of the best-fit linear operator $\mathbf{A}$.

#### Continuous-time Eigenvalues

$$\omega_j = \frac{\ln \lambda_j}{\Delta t}$$

- Real part: growth/decay rate
- Imaginary part: oscillation frequency

#### Spectral Radius

$$\rho = \max_j |\lambda_j|$$

**Interpretation:**

| ρ | Meaning |
|---|---------|
| < 1 | Stable (all modes decay) |
| > 1 | Unstable (at least one mode grows) |
| ≈ 1 | Marginally stable / persistent oscillation |

**Minimum samples:** 63

---

### Transfer Entropy

**Measures:** Directed information flow.

**Question:** How much does X's past reduce uncertainty about Y's future, beyond Y's own past?

**Why it matters:** Unlike mutual information (symmetric), transfer entropy is asymmetric. It quantifies the directed flow of information from source to target, revealing causal influence in information-theoretic terms.

#### Formula

$$T_{X \to Y} = H(Y_t | Y_{t-1}^{(k)}) - H(Y_t | Y_{t-1}^{(k)}, X_{t-1}^{(l)})$$

$$= \sum p(y_t, y_{t-1}^{(k)}, x_{t-1}^{(l)}) \log \frac{p(y_t | y_{t-1}^{(k)}, x_{t-1}^{(l)})}{p(y_t | y_{t-1}^{(k)})}$$

where:
- $k$ = embedding dimension for Y
- $l$ = embedding dimension for X

#### Net Transfer Entropy

$$NET_{X \to Y} = T_{X \to Y} - T_{Y \to X}$$

Positive means net information flow from X to Y.

**Minimum samples:** 100

---

## Engine Summary Table

### Vector Engines (Per-Indicator)

| Engine | Measures | Key Formula | Min Samples |
|--------|----------|-------------|-------------|
| Hurst | Memory | $H = \log(R/S) / \log(n)$ | 100 |
| Entropy | Complexity | $H = -\sum p \log p$ | 50 |
| GARCH | Vol clustering | $\sigma_t^2 = \omega + \alpha\epsilon^2 + \beta\sigma^2$ | 63 |
| Wavelet | Multi-scale | $W(a,b) = \int x(t)\psi^*((t-b)/a)dt$ | 64 |
| Spectral | Frequencies | $P(f) = |X(f)|^2$ | 64 |
| Lyapunov | Chaos | $\lambda = \lim \frac{1}{t}\ln\frac{\|\delta x(t)\|}{\|\delta x(0)\|}$ | 200 |
| RQA | Recurrence | $R_{i,j} = \Theta(\epsilon - \|x_i - x_j\|)$ | 100 |

### Geometry Engines (Multi-Indicator, Static)

| Engine | Measures | Key Formula | Min Indicators |
|--------|----------|-------------|----------------|
| PCA | Shared variance | $\Sigma v = \lambda v$ | 3 |
| Clustering | Groupings | $s = (b-a)/\max(a,b)$ | 4 |
| Distance | Dissimilarity | $d = \sqrt{\sum(x_i-x_j)^2}$ | 2 |
| Mutual Info | Dependence | $I = H(X) + H(Y) - H(X,Y)$ | 2 |
| Copula | Tail dependence | $\lambda_U = \lim P(V>u|U>u)$ | 2 |
| MST | Connectivity | $\min \sum d_{ij}$ | 3 |
| LOF | Outliers | $LOF = \text{avg}(lrd_{neighbors}/lrd)$ | 4 |
| Convex Hull | Extent | Boundary + centrality | 3 |

### State Engines (Multi-Indicator, Temporal)

| Engine | Measures | Key Formula | Min Samples |
|--------|----------|-------------|-------------|
| Granger | Predictive causality | F-test on lagged regression | 50 |
| Cross-Corr | Lead/lag | $\rho(k) = \text{Cov}(x_t, y_{t+k})$ | 50 |
| Cointegration | Equilibrium | ADF on residuals | 100 |
| DTW | Shape similarity | $D(i,j) = d + \min(D_{neighbors})$ | 30 |
| DMD | Dynamic modes | $X' = AX$, eigendecomposition | 63 |
| Transfer Entropy | Info flow | $T = H(Y|Y_{past}) - H(Y|Y_{past},X_{past})$ | 100 |

---

## Design Philosophy

### Vector → Geometry → State

```
Raw Observations
      │
      ▼
┌─────────────────┐
│ VECTOR ENGINES  │  "What is each series?"
│ (per-indicator) │
└────────┬────────┘
         │ Behavioral Descriptors
         ▼
┌─────────────────┐
│ GEOMETRY ENGINES│  "How do they relate now?"
│ (static structure)│
└────────┬────────┘
         │ Structural Relationships
         ▼
┌─────────────────┐
│ STATE ENGINES   │  "How do they evolve together?"
│ (temporal dynamics)│
└─────────────────┘
```

### Why Three Tiers?

1. **Vector engines** produce behavioral descriptors that compress each signal topology into a fixed-dimensional representation.

2. **Geometry engines** analyze relationships between these descriptors without needing the original time ordering — they see indicators as points in behavioral space.

3. **State engines** require the full temporal structure to analyze causality, lead-lag relationships, and dynamic coupling that geometry cannot capture.

---

## References

- Bandt & Pompe (2002). Permutation entropy: A natural complexity measure for signal topology.
- Bollerslev (1986). Generalized autoregressive conditional heteroskedasticity.
- Engle & Granger (1987). Co-integration and error correction.
- Mallat (1989). A theory for multiresolution signal decomposition.
- Marwan et al. (2007). Recurrence plots for the analysis of complex systems.
- Rosenstein et al. (1993). A practical method for calculating largest Lyapunov exponents.
- Schreiber (2000). Measuring information transfer.
- Schmid (2010). Dynamic mode decomposition of numerical and experimental data.
- Sklar (1959). Fonctions de répartition à n dimensions et leurs marges.
- Takens (1981). Detecting strange attractors in turbulence.

---

*PRISM Core — Measurement-First Analysis*
