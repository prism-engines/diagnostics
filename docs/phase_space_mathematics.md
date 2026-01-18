# Phase Space Mathematics and PRISM

## Overview

This document covers the classical mathematics of phase space and dynamical systems, then maps those concepts to what PRISM does - highlighting where they align and where PRISM diverges.

---

## Part 1: Classical Phase Space Math

### 1.1 The State Vector

In physics, a **state vector** captures everything you need to know about a system at one instant.

For a single particle moving in one dimension:

```
State vector: x = [position, velocity] = [q, p]
```

For N particles in 3D space:

```
State vector: x = [q₁, q₂, ..., q₃ₙ, p₁, p₂, ..., p₃ₙ]
```

That's 6N dimensions - 3 position coordinates and 3 momentum coordinates per particle.

**The key property:** Given the state vector at time t, the laws of physics determine the state at time t+1. The state vector contains *complete information* about the system's future evolution.

### 1.2 Phase Space

**Phase space** is the set of all possible state vectors - every configuration the system could be in.

For a pendulum:
- Position θ: angle from vertical (-π to π)
- Momentum p: angular momentum

Phase space is a 2D plane where every point represents a possible pendulum state.

```
        p (momentum)
        ^
        |    ___
        |   /   \      ← Oscillating (ellipse)
        |  |     |
   -----+--+-----+----- → θ (angle)
        |  |     |
        |   \___/
        |
```

### 1.3 Trajectories and Flow

A **trajectory** is the path a system takes through phase space over time.

Mathematically, it's the solution to differential equations:

```
dq/dt = ∂H/∂p
dp/dt = -∂H/∂q
```

Where H is the Hamiltonian (total energy). These are **Hamilton's equations**.

The collection of all trajectories defines a **flow** - like water currents, showing how any starting point evolves.

### 1.4 Attractors

An **attractor** is a region of phase space where trajectories tend to end up.

Types:
- **Fixed point**: System settles to one state (e.g., pendulum at rest)
- **Limit cycle**: System oscillates forever (e.g., heartbeat)
- **Strange attractor**: Chaotic motion confined to a complex structure (e.g., weather)

```
Fixed Point          Limit Cycle          Strange Attractor
    •                   ___                    ~~~~~
   /|\                 /   \                  ~~~~~~~
  / | \               |     |                ~~~°~~~~
 /  •  \               \___/                  ~~~~~
```

### 1.5 Lyapunov Exponents

**Lyapunov exponents** measure how fast nearby trajectories diverge.

```
λ = lim(t→∞) (1/t) × ln(|δx(t)| / |δx(0)|)
```

Where:
- δx(0) is initial separation between two nearby points
- δx(t) is separation after time t

Interpretation:
- λ > 0: Chaotic (trajectories diverge exponentially)
- λ = 0: Neutral (trajectories stay parallel)
- λ < 0: Stable (trajectories converge)

---

## Part 2: Reconstructing Phase Space from Data

### 2.1 The Problem

In physics, you know the state variables (position, momentum).

In real data, you often observe only ONE variable (e.g., temperature, vibration) but the underlying system has many hidden dimensions.

**Question:** Can you recover the phase space structure from a single observed signal topology?

### 2.2 Takens' Embedding Theorem (1981)

**Takens' theorem** says: Yes, under certain conditions.

Given a signal topology {x(t)}, construct **delay vectors**:

```
v(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
```

Where:
- τ = time delay (lag)
- d = embedding dimension

**The theorem states:** If d > 2D (where D is the true dimension of the attractor), the reconstructed phase space has the same topological structure as the original.

### 2.3 Example: Reconstructing a Pendulum

Suppose you only observe the angle θ(t), not the momentum.

Create delay vectors with d=2, τ=T/4 (quarter period):

```
v(t) = [θ(t), θ(t - T/4)]
```

The plot of θ(t) vs θ(t - T/4) recovers an ellipse - the same structure as the true θ vs p phase space.

```
Original Phase Space        Reconstructed (Delay Embedding)
      p                           θ(t-τ)
      ^                              ^
      |   ___                        |   ___
      |  /   \                       |  /   \
      | |     |                      | |     |
------+-------→ θ            --------+-------→ θ(t)
      | |     |                      | |     |
      |  \___/                       |  \___/
```

### 2.4 Choosing Parameters

**Embedding dimension (d):** Use "false nearest neighbors" algorithm - increase d until the fraction of false neighbors drops to zero.

**Time delay (τ):** Use first minimum of mutual information or autocorrelation function.

---

## Part 3: Comparing to PRISM

### 3.1 Classical Phase Space vs. PRISM Behavioral Space

| Aspect | Classical Phase Space | PRISM Behavioral Space |
|--------|----------------------|------------------------|
| **Dimensions** | Physical quantities (position, momentum) | Behavioral descriptors (Hurst, entropy) |
| **Construction** | From first principles or delay embedding | From heterogeneous mathematical engines |
| **State vector** | [q₁, q₂, ..., p₁, p₂, ...] | [memory, complexity, cyclicality, ...] |
| **Trajectories** | Solutions to differential equations | Rolling window behavioral snapshots |
| **Goal** | Understand dynamical evolution | Understand structural relationships |

### 3.2 Key Difference: What Are the Dimensions?

**Takens embedding:**
```
v(t) = [x(t), x(t-τ), x(t-2τ), ...]
```
Dimensions are *lagged copies of the same variable*.

**PRISM behavioral vector:**
```
v = [Hurst, SampleEntropy, WaveletEnergy, GARCH_vol, ...]
```
Dimensions are *different mathematical characterizations of behavior*.

**This is fundamentally different.** Takens reconstructs the *same* phase space from limited observations. PRISM constructs a *new* space from behavioral measurements.

### 3.3 What PRISM Does That's Different

Classical approach:
> "The system has hidden dimensions. Let's recover them from the signal topology."

PRISM approach:
> "Let's define new dimensions based on behavioral properties. These become the space."

PRISM isn't reconstructing a hidden physical phase space. It's **constructing** a behavioral phase space where the dimensions are chosen to be interpretable and meaningful.

---

## Part 4: The Mathematics in PRISM's Engines

### 4.1 Vector Engines (Behavioral Descriptors)

These compute the coordinates in behavioral space.

#### Hurst Exponent (Memory Dimension)

Measures long-range dependence / persistence.

**R/S Analysis Method:**

For a signal topology of length n, divide into subseries of length τ.

For each subseries:
1. Compute mean: m = (1/τ) Σ xᵢ
2. Compute cumulative deviation: Yₜ = Σᵢ₌₁ᵗ (xᵢ - m)
3. Compute range: R(τ) = max(Yₜ) - min(Yₜ)
4. Compute standard deviation: S(τ)

The Hurst exponent H comes from:

```
E[R(τ)/S(τ)] ∝ τᴴ
```

Or equivalently: log(R/S) = H × log(τ) + c

**Interpretation:**
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trending)
- H < 0.5: Anti-persistent (mean-reverting)

#### Sample Entropy (Complexity Dimension)

Measures irregularity / unpredictability.

**Algorithm:**

Given signal topology {x₁, x₂, ..., xₙ}, embedding dimension m, tolerance r:

1. Form template vectors: uᵢᵐ = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁]

2. Count matches: Bᵢᵐ = (number of j where d(uᵢᵐ, uⱼᵐ) < r) / (N-m-1)

3. Compute:
   - Bᵐ = (1/(N-m)) Σᵢ Bᵢᵐ
   - Bᵐ⁺¹ = same thing with m+1

4. Sample Entropy = -ln(Bᵐ⁺¹ / Bᵐ)

**Interpretation:**
- Higher entropy = more complex/irregular
- Lower entropy = more regular/predictable

#### Wavelet Energy (Cyclicality Dimension)

Decomposes signal into frequency bands.

**Discrete Wavelet Transform:**

```
cⱼ,ₖ = Σₙ x(n) × ψⱼ,ₖ(n)
```

Where ψⱼ,ₖ is a wavelet at scale j and position k.

Energy at scale j:
```
Eⱼ = Σₖ |cⱼ,ₖ|²
```

Relative energy distribution reveals dominant time scales.

**Interpretation:**
- Energy concentrated at long scales = slow cycles
- Energy at short scales = fast oscillations
- Flat distribution = no dominant periodicity

#### GARCH Volatility (Tail/Risk Dimension)

Models time-varying variance.

**GARCH(1,1) model:**

```
rₜ = μ + εₜ
εₜ = σₜ × zₜ,  where zₜ ~ N(0,1)
σₜ² = ω + α×εₜ₋₁² + β×σₜ₋₁²
```

Parameters:
- ω: baseline variance
- α: shock impact (how much recent squared errors matter)
- β: persistence (how much past variance matters)

**Interpretation:**
- High α: Volatility spikes sharply after shocks
- High β: Volatility clusters persist
- α + β close to 1: Long memory in volatility

### 4.2 Geometry Engines (Structural Relationships)

These compute relationships between points in behavioral space.

#### Euclidean Distance

Most basic measure of separation.

```
d(x, y) = √[Σᵢ (xᵢ - yᵢ)²]
```

For two indicators with behavioral vectors x and y, this measures "how different are their behavioral signatures?"

#### Mahalanobis Distance

Accounts for correlations between dimensions.

```
d(x, y) = √[(x - y)ᵀ × Σ⁻¹ × (x - y)]
```

Where Σ is the covariance matrix of the behavioral dimensions.

**Why it matters:** If memory and complexity are correlated, Euclidean distance over-counts their contribution. Mahalanobis corrects for this.

#### Cosine Similarity

Measures angle between vectors, ignoring magnitude.

```
cos(θ) = (x · y) / (|x| × |y|) = Σᵢ(xᵢyᵢ) / (√Σᵢxᵢ² × √Σᵢyᵢ²)
```

**Interpretation:**
- 1: Same direction (proportional behavioral profiles)
- 0: Orthogonal (unrelated profiles)
- -1: Opposite directions

#### Principal Component Analysis (PCA)

Finds the main axes of variation in behavioral space.

**Computation:**
1. Center the data: X̃ = X - mean(X)
2. Compute covariance: C = X̃ᵀX̃ / (n-1)
3. Eigendecomposition: C = VΛVᵀ
4. Principal components are columns of V
5. Eigenvalues in Λ give variance explained

**Interpretation:**
- PC1: Direction of maximum variance in behavioral space
- Eigenvalue ratio: How much structure is captured by top components
- Loadings: Which behavioral dimensions contribute to each PC

#### Dynamic Time Warping (DTW)

Measures similarity allowing for time shifts.

**Original application:** Compare signal topology that may be stretched/compressed.

**In PRISM's behavioral space:** Compare trajectories through behavioral space, allowing for different speeds.

**Algorithm:**

Given two sequences X = [x₁, ..., xₙ] and Y = [y₁, ..., yₘ]:

1. Build cost matrix: C(i,j) = d(xᵢ, yⱼ)

2. Build accumulated cost matrix:
```
D(i,j) = C(i,j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1))
```

3. DTW distance = D(n, m)

**In behavioral context:** Two indicators might traverse similar behavioral regions but at different speeds. DTW captures this.

---

## Part 5: What's Similar, What's Different

### 5.1 Shared Mathematical DNA

| Classical Tool | PRISM Equivalent | Relationship |
|---------------|------------------|--------------|
| State vector | Behavioral vector | Same structure, different dimensions |
| Phase space | Behavioral space | Same concept, constructed vs. reconstructed |
| Trajectory | Rolling window path | Same idea - motion through space |
| Attractor | Behavioral cluster | Regions where indicators congregate |
| Lyapunov exponent | PRISM doesn't compute this directly | Could be added |

### 5.2 Key Mathematical Differences

**1. Dimension Construction**

Classical:
```
Dimensions = [position, momentum] or delay embeddings
Derived from physics or Takens theorem
```

PRISM:
```
Dimensions = [Hurst, Entropy, Wavelet, GARCH, ...]
Chosen for interpretability and coverage
```

**2. Heterogeneous Measurements**

Classical phase space has dimensions with the same units (or conjugate pairs like position/momentum).

PRISM combines dimensionally different quantities:
- Hurst is unitless (0 to 1)
- Entropy is in bits
- GARCH parameters have various scales

This requires normalization before computing geometry.

**3. Multiple Entities**

Classical: Usually one system moving through phase space.

PRISM: Many indicators, each with a position in behavioral space. The geometry is about *relationships between entities*, not just evolution of one.

### 5.3 What PRISM Could Add from Classical Theory

#### Lyapunov Exponents on Behavioral Trajectories

Compute how fast nearby behavioral trajectories diverge:

```python
def behavioral_lyapunov(trajectory, delta_t, epsilon=1e-6):
    """
    trajectory: array of shape (T, D) - behavioral vectors over time
    Returns approximate largest Lyapunov exponent
    """
    T, D = trajectory.shape
    
    # Find nearest neighbor at each time
    divergence_rates = []
    for t in range(T - delta_t):
        # Current point
        x_t = trajectory[t]
        
        # Find nearest neighbor (excluding temporal neighbors)
        distances = np.linalg.norm(trajectory - x_t, axis=1)
        distances[max(0,t-10):t+10] = np.inf  # Exclude temporal neighbors
        nn_idx = np.argmin(distances)
        
        # Initial separation
        d0 = distances[nn_idx]
        
        # Separation after delta_t
        if nn_idx + delta_t < T:
            d1 = np.linalg.norm(trajectory[t + delta_t] - trajectory[nn_idx + delta_t])
            
            if d0 > epsilon and d1 > epsilon:
                divergence_rates.append(np.log(d1 / d0) / delta_t)
    
    return np.mean(divergence_rates)
```

**Interpretation:** Positive Lyapunov exponent in behavioral space would indicate chaotic behavioral dynamics - small differences in current behavioral state lead to large differences later.

#### Recurrence Analysis

Track when the system returns to similar behavioral states:

```python
def recurrence_matrix(trajectory, threshold):
    """
    Build recurrence matrix for behavioral trajectory
    R[i,j] = 1 if ||x_i - x_j|| < threshold
    """
    T = len(trajectory)
    R = np.zeros((T, T))
    
    for i in range(T):
        for j in range(T):
            if np.linalg.norm(trajectory[i] - trajectory[j]) < threshold:
                R[i, j] = 1
    
    return R
```

Recurrence plots reveal:
- Diagonal lines: Deterministic dynamics
- Vertical/horizontal lines: Laminar states (system stuck in a region)
- Isolated points: Chaotic behavior

---

## Part 6: Proposed Mathematical Extensions for PRISM

### 6.1 Behavioral Velocity and Acceleration

If you have rolling behavioral vectors, compute derivatives:

```
Velocity: v(t) = [b(t+Δt) - b(t)] / Δt

Acceleration: a(t) = [v(t+Δt) - v(t)] / Δt
```

**New metrics:**
- |v|: Speed of behavioral change
- v · v_prev / (|v||v_prev|): Consistency of direction
- |a|: Rate of behavioral regime change

### 6.2 Curvature of Behavioral Trajectories

How sharply is the trajectory bending?

```
κ = |v × a| / |v|³
```

High curvature = sudden behavioral shifts
Low curvature = smooth evolution

### 6.3 Behavioral Divergence Between Indicators

Track how two indicators' behavioral trajectories separate over time:

```
D(t) = ||b_A(t) - b_B(t)||

dD/dt = rate of divergence/convergence
```

**Crisis indicator:** If normally-stable pairs suddenly diverge in behavioral space, something structural is changing.

### 6.4 Phase Space Volume (Behavioral Dispersion)

For a collection of indicators, track the volume of behavioral space they occupy:

```
Volume ≈ det(Covariance matrix of behavioral vectors)^(1/2)
```

Or use convex hull volume in lower-dimensional projection.

**Interpretation:**
- Contracting volume: Indicators converging to similar behavior
- Expanding volume: Indicators diverging, heterogeneous behavior
- Pre-crisis pattern might show contraction followed by expansion

---

## Part 7: Summary Table

| Math Concept | Formula/Method | PRISM Application |
|--------------|----------------|-------------------|
| State vector | x = [q₁, ..., pₙ] | Behavioral vector = [H, SE, W, G, ...] |
| Trajectory | x(t) = solution to dx/dt = f(x) | Rolling window behavioral snapshots |
| Distance | d = √[Σ(xᵢ-yᵢ)²] | Behavioral similarity between indicators |
| PCA | Eigendecomposition of covariance | Main axes of behavioral variation |
| DTW | Optimal alignment cost | Trajectory similarity with time flexibility |
| Lyapunov | λ = lim ln(δx(t)/δx(0))/t | Stability of behavioral dynamics |
| Recurrence | R[i,j] = 1 if d(xᵢ,xⱼ) < ε | When system revisits behavioral states |
| Curvature | κ = |v × a| / |v|³ | Sharpness of behavioral change |

---

## Part 8: The Conceptual Leap

Classical phase space analysis asks:
> "Given this dynamical system, what is its phase space structure?"

PRISM asks:
> "Given these signal topology, let's construct a behavioral phase space and analyze its geometry."

The math tools (distance, PCA, trajectories) are borrowed from classical theory.

The **innovation** is:
1. Defining dimensions as behavioral descriptors (not physical or delay-embedded)
2. Placing multiple entities in the same space
3. Computing geometry between entities, not just evolution of one
4. Using that geometry to detect structural change

This is why the seismology data worked without modification - the behavioral dimensions (memory, complexity, etc.) are domain-agnostic. A Hurst exponent measures persistence whether the underlying data is bearing vibration or earthquake intervals.

---

*Document created for PRISM project*
*Bridging classical dynamical systems mathematics with behavioral space analysis*
