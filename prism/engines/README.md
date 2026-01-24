# PRISM Engines - Mathematical Reference

**For the 2am grad student who needs to verify calculations are correct.**

All formulas include step-by-step derivations. If the code doesn't match these formulas, the code is wrong.

---

## Table of Contents

1. [Stage 1: Signal Typology](#stage-1-signal-typology) - Single signal → behavioral metrics
2. [Stage 2: Structural Geometry](#stage-2-structural-geometry) - Signals → pairwise relationships
3. [Stage 3: Dynamical Systems](#stage-3-dynamical-systems) - Geometry evolution over time
4. [Stage 4: Causal Mechanics](#stage-4-causal-mechanics) - Causal relationships

---

# Stage 1: Signal Typology

These engines analyze a **single signal** to extract behavioral metrics.

---

## 1.1 Hurst Exponent (R/S Method)

**File:** `memory/hurst_rs.py`
**Windowed:** YES
**What it measures:** Long-range dependence. Does the past predict the future?

### Formula

$$H = \frac{\log(R/S)}{\log(n)}$$

Where:
- $H$ = Hurst exponent (what we want)
- $R$ = Range of cumulative deviations
- $S$ = Standard deviation
- $n$ = Window size

### Step-by-Step Derivation

**Given:** A time series $x_1, x_2, ..., x_n$

**Step 1: Compute the mean**
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

*Example: For [2, 4, 6, 8, 10], mean = (2+4+6+8+10)/5 = 6*

**Step 2: Compute deviations from mean**
$$d_i = x_i - \bar{x}$$

*Example: d = [-4, -2, 0, 2, 4]*

**Step 3: Compute cumulative deviations**
$$Y_t = \sum_{i=1}^{t} d_i$$

*Example: Y = [-4, -6, -6, -4, 0]*

**Step 4: Compute Range**
$$R = \max(Y) - \min(Y)$$

*Example: R = 0 - (-6) = 6*

**Step 5: Compute Standard Deviation**
$$S = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

*Example: S = sqrt((16+4+0+4+16)/5) = sqrt(8) ≈ 2.83*

**Step 6: Compute R/S ratio**
$$R/S = \frac{6}{2.83} \approx 2.12$$

**Step 7: Estimate H via log-log regression**

Do this for multiple window sizes, then:
$$\log(R/S) = H \cdot \log(n) + c$$

Fit a line. The slope is H.

### How to Verify

```python
# Quick sanity check
import numpy as np

x = np.array([2, 4, 6, 8, 10])
mean = np.mean(x)                    # 6.0
devs = x - mean                       # [-4, -2, 0, 2, 4]
cumsum = np.cumsum(devs)              # [-4, -6, -6, -4, 0]
R = np.max(cumsum) - np.min(cumsum)   # 6.0
S = np.std(x, ddof=0)                 # 2.83
RS = R / S                            # 2.12
```

### Interpretation

| H Value | Meaning | In Plain English |
|---------|---------|------------------|
| H < 0.5 | Anti-persistent | Goes up → likely goes down next |
| H = 0.5 | Random walk | Past tells you nothing |
| H > 0.5 | Persistent | Goes up → likely keeps going up |

---

## 1.2 Hurst Exponent (DFA Method)

**File:** `memory/hurst_dfa.py`
**Windowed:** YES
**What it measures:** Same as R/S, but robust to non-stationarity (trends don't fool it)

### Formula

$$F(n) \propto n^H$$

Where $F(n)$ is the fluctuation function (RMS of detrended segments).

### Step-by-Step Derivation

**Given:** A time series $x_1, x_2, ..., x_N$

**Step 1: Integrate the series**
$$y_k = \sum_{i=1}^{k} (x_i - \bar{x})$$

*This converts the series to a random walk*

**Step 2: Divide into windows of size n**

For window size n=10 on a series of length 100, you get 10 windows.

**Step 3: For each window, fit a line and detrend**
$$y_{fit} = a \cdot t + b$$
$$y_{detrended} = y - y_{fit}$$

**Step 4: Compute RMS fluctuation for window**
$$F_{window} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} y_{detrended,i}^2}$$

**Step 5: Average across all windows**
$$F(n) = \sqrt{\frac{1}{\#windows} \sum_{w} F_w^2}$$

**Step 6: Repeat for multiple window sizes**

Compute F(n) for n = 10, 15, 22, 33, 50, 75, ...

**Step 7: Log-log regression**
$$\log F(n) = H \cdot \log(n) + c$$

The slope is H.

### How to Verify

```python
import numpy as np

def dfa_fluctuation(y, window_size):
    """Compute F(n) for one window size"""
    n_windows = len(y) // window_size
    f2_list = []

    for i in range(n_windows):
        segment = y[i*window_size : (i+1)*window_size]

        # Fit and remove linear trend
        x = np.arange(window_size)
        slope, intercept = np.polyfit(x, segment, 1)
        trend = slope * x + intercept
        detrended = segment - trend

        # RMS
        f2 = np.mean(detrended**2)
        f2_list.append(f2)

    return np.sqrt(np.mean(f2_list))
```

### Why DFA > R/S

R/S gets confused by trends. DFA removes them. Use DFA for real-world data.

---

## 1.3 ACF Decay

**File:** `memory/acf_decay.py`
**Windowed:** YES
**What it measures:** How fast does autocorrelation die? Fast decay = short memory.

### Formula

$$\rho(k) = \frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}{\sum_{t=1}^{n} (x_t - \bar{x})^2}$$

Where:
- $\rho(k)$ = autocorrelation at lag k
- $k$ = lag (1, 2, 3, ...)

### Step-by-Step Derivation

**Given:** Time series $x_1, x_2, ..., x_n$

**Step 1: Compute mean**
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Step 2: Compute variance (denominator)**
$$\text{var} = \sum_{t=1}^{n} (x_t - \bar{x})^2$$

**Step 3: For each lag k, compute covariance (numerator)**
$$\text{cov}(k) = \sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})$$

**Step 4: Divide**
$$\rho(k) = \frac{\text{cov}(k)}{\text{var}}$$

**Step 5: Fit exponential decay**
$$\rho(k) \approx e^{-k/\tau}$$

Take log: $\log(\rho(k)) = -k/\tau$

Slope of log(ACF) vs k gives decay rate.

### How to Verify

```python
import numpy as np

x = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2])
mean = np.mean(x)
var = np.sum((x - mean)**2)

# ACF at lag 1
lag = 1
cov = np.sum((x[:-lag] - mean) * (x[lag:] - mean))
acf_1 = cov / var
print(f"ACF(1) = {acf_1}")
```

### Interpretation

| Decay Rate | Meaning |
|------------|---------|
| Fast (τ small) | Short memory, quickly forgets |
| Slow (τ large) | Long memory, past matters |

---

## 1.4 Permutation Entropy

**File:** `information/permutation_entropy.py`
**Windowed:** YES
**What it measures:** Complexity via ordinal patterns. How unpredictable is the order?

### Formula

$$H_p = -\sum_{i=1}^{m!} p_i \log_2(p_i)$$

Where:
- $m$ = embedding dimension (typically 3-7)
- $p_i$ = probability of pattern i
- $m!$ = number of possible patterns

### Step-by-Step Derivation

**Given:** Time series [4, 7, 9, 10, 6, 11, 3]

**Step 1: Choose embedding dimension m**

Let's use m = 3 (look at 3 consecutive values)

**Step 2: Extract all windows of size m**

| Window | Values | Rank Order |
|--------|--------|------------|
| 1 | [4, 7, 9] | [0, 1, 2] (ascending) |
| 2 | [7, 9, 10] | [0, 1, 2] (ascending) |
| 3 | [9, 10, 6] | [1, 2, 0] (middle, high, low) |
| 4 | [10, 6, 11] | [1, 0, 2] |
| 5 | [6, 11, 3] | [1, 2, 0] |

**Step 3: Count pattern frequencies**

| Pattern | Count | Probability |
|---------|-------|-------------|
| [0,1,2] | 2 | 2/5 = 0.4 |
| [1,2,0] | 2 | 2/5 = 0.4 |
| [1,0,2] | 1 | 1/5 = 0.2 |

**Step 4: Compute entropy**
$$H_p = -(0.4 \log_2 0.4 + 0.4 \log_2 0.4 + 0.2 \log_2 0.2)$$
$$H_p = -(0.4 \times -1.32 + 0.4 \times -1.32 + 0.2 \times -2.32)$$
$$H_p = -(-0.53 - 0.53 - 0.46) = 1.52$$

**Step 5: Normalize (optional)**
$$H_{norm} = \frac{H_p}{\log_2(m!)} = \frac{1.52}{\log_2(6)} = \frac{1.52}{2.58} = 0.59$$

### How to Verify

```python
import numpy as np
from collections import Counter

def permutation_entropy(x, m=3):
    n = len(x)
    patterns = []

    for i in range(n - m + 1):
        window = x[i:i+m]
        # Get rank order (argsort of argsort)
        pattern = tuple(np.argsort(np.argsort(window)))
        patterns.append(pattern)

    # Count frequencies
    counts = Counter(patterns)
    total = len(patterns)

    # Compute entropy
    H = 0
    for count in counts.values():
        p = count / total
        H -= p * np.log2(p)

    return H

x = np.array([4, 7, 9, 10, 6, 11, 3])
print(f"PE = {permutation_entropy(x, m=3)}")
```

### Interpretation

| H_norm | Meaning |
|--------|---------|
| ~0 | Perfectly predictable (always same pattern) |
| ~0.5 | Some structure |
| ~1 | Maximum complexity (all patterns equally likely) |

---

## 1.5 Sample Entropy

**File:** `information/sample_entropy.py`
**Windowed:** YES
**What it measures:** Regularity/predictability. Low = regular, high = random.

### Formula

$$SampEn(m, r) = -\ln\frac{A}{B}$$

Where:
- $m$ = embedding dimension
- $r$ = tolerance (typically 0.2 × std)
- $A$ = count of matching templates of length m+1
- $B$ = count of matching templates of length m

### Step-by-Step Derivation

**Given:** Time series $x = [1, 2, 1, 2, 1, 2, 1]$, m=2, r=0.5

**Step 1: Create templates of length m=2**

| Template | Values |
|----------|--------|
| T1 | [1, 2] |
| T2 | [2, 1] |
| T3 | [1, 2] |
| T4 | [2, 1] |
| T5 | [1, 2] |

**Step 2: Count matches for m=2 (B)**

Two templates match if ALL elements are within r of each other.

T1=[1,2] vs T3=[1,2]: |1-1|=0 ≤ 0.5, |2-2|=0 ≤ 0.5 → MATCH
T1=[1,2] vs T5=[1,2]: MATCH
...

Count all pairs (excluding self-matches): B = count

**Step 3: Create templates of length m+1=3**

| Template | Values |
|----------|--------|
| T1 | [1, 2, 1] |
| T2 | [2, 1, 2] |
| T3 | [1, 2, 1] |
| T4 | [2, 1, 2] |

**Step 4: Count matches for m+1=3 (A)**

Same process, but with longer templates.

**Step 5: Compute Sample Entropy**
$$SampEn = -\ln\frac{A}{B}$$

### How to Verify

```python
import numpy as np

def sample_entropy(x, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(x)

    n = len(x)

    def count_matches(template_len):
        templates = []
        for i in range(n - template_len + 1):
            templates.append(x[i:i+template_len])

        count = 0
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                # Check if all elements match within r
                if np.all(np.abs(templates[i] - templates[j]) <= r):
                    count += 1
        return count

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0 or A == 0:
        return np.nan

    return -np.log(A / B)

x = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
print(f"SampEn = {sample_entropy(x)}")  # Should be low (regular)
```

### Interpretation

| SampEn | Meaning |
|--------|---------|
| ~0 | Highly regular/predictable |
| ~1-2 | Normal complexity |
| >2 | High irregularity/randomness |

---

## 1.6 GARCH(1,1) Volatility

**File:** `volatility/garch.py`
**Windowed:** YES
**What it measures:** Volatility clustering. Big moves followed by big moves?

### Formula

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Where:
- $\sigma_t^2$ = conditional variance at time t
- $\omega$ = constant (baseline volatility)
- $\alpha$ = shock impact (how much yesterday's surprise matters)
- $\beta$ = persistence (how much yesterday's volatility matters)
- $\epsilon_{t-1}$ = yesterday's shock (return - expected return)

### Step-by-Step Derivation

**Given:** Returns $r_1, r_2, ..., r_n$

**Step 1: Initialize**
$$\sigma_1^2 = \text{Var}(r) \quad \text{(unconditional variance)}$$

**Step 2: For each t > 1, compute conditional variance**
$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$

*Note: Assuming mean return = 0, so $\epsilon_t = r_t$*

**Step 3: Example calculation**

Let $\omega=0.01$, $\alpha=0.1$, $\beta=0.8$, $\sigma_1^2=0.04$

| t | $r_{t-1}$ | $r_{t-1}^2$ | $\sigma_{t-1}^2$ | $\sigma_t^2$ |
|---|-----------|-------------|------------------|--------------|
| 2 | 0.05 | 0.0025 | 0.04 | 0.01 + 0.1×0.0025 + 0.8×0.04 = 0.04225 |
| 3 | -0.10 | 0.01 | 0.04225 | 0.01 + 0.1×0.01 + 0.8×0.04225 = 0.0448 |
| 4 | 0.02 | 0.0004 | 0.0448 | 0.01 + 0.1×0.0004 + 0.8×0.0448 = 0.04624 |

**Step 4: Estimate parameters via MLE**

Maximize:
$$\mathcal{L} = -\frac{1}{2} \sum_{t=1}^{n} \left( \log(\sigma_t^2) + \frac{r_t^2}{\sigma_t^2} \right)$$

### Key Constraints

$$\alpha + \beta < 1 \quad \text{(stationarity)}$$
$$\omega > 0, \alpha \geq 0, \beta \geq 0$$

### How to Verify

```python
import numpy as np

def garch_variance(returns, omega, alpha, beta):
    """Compute GARCH(1,1) conditional variance series"""
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)  # Initialize

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

    return sigma2

# Example
returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01])
omega, alpha, beta = 0.0001, 0.1, 0.85
sigma2 = garch_variance(returns, omega, alpha, beta)
print(f"Conditional volatility: {np.sqrt(sigma2)}")
```

### Interpretation

| Parameter | Meaning |
|-----------|---------|
| High α | Shocks have big immediate impact |
| High β | Volatility is persistent |
| α + β ≈ 1 | Integrated GARCH (volatility doesn't mean-revert) |

---

## 1.7 Realized Volatility

**File:** `volatility/realized_vol.py`
**Windowed:** YES
**What it measures:** Actual observed volatility (not modeled).

### Formula

$$RV = \sqrt{\sum_{t=1}^{n} r_t^2}$$

Where $r_t = x_t - x_{t-1}$ (or log returns: $r_t = \ln(x_t/x_{t-1})$)

### Step-by-Step Derivation

**Given:** Prices $x_1, x_2, ..., x_n$

**Step 1: Compute returns**
$$r_t = x_t - x_{t-1}$$

*Example: prices [100, 102, 99, 103] → returns [2, -3, 4]*

**Step 2: Square the returns**
$$r^2 = [4, 9, 16]$$

**Step 3: Sum squared returns**
$$\sum r^2 = 4 + 9 + 16 = 29$$

**Step 4: Take square root**
$$RV = \sqrt{29} \approx 5.39$$

### Annualization (Optional)

If you have daily data and want annual volatility:
$$RV_{annual} = RV_{daily} \times \sqrt{252}$$

*(252 = trading days per year)*

### How to Verify

```python
import numpy as np

prices = np.array([100, 102, 99, 103, 101, 105])
returns = np.diff(prices)
realized_vol = np.sqrt(np.sum(returns**2))
print(f"RV = {realized_vol}")
```

### Interpretation

Higher RV = more volatile period. Compare RV to GARCH prediction to see if model is accurate.

---

## 1.8 Lyapunov Exponent

**File:** `dynamics/lyapunov.py`
**Windowed:** NO (but computationally expensive)
**What it measures:** Chaos. How fast do nearby trajectories diverge?

### Formula

$$\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{d(t)}{d(0)}$$

Where:
- $\lambda$ = largest Lyapunov exponent
- $d(0)$ = initial distance between nearby points
- $d(t)$ = distance after time t

### Step-by-Step Derivation (Wolf Algorithm Simplified)

**Given:** Time series $x_1, x_2, ..., x_n$

**Step 1: Embed in phase space**

Create vectors: $\mathbf{v}_i = [x_i, x_{i+\tau}, x_{i+2\tau}, ..., x_{i+(m-1)\tau}]$

Where:
- $m$ = embedding dimension
- $\tau$ = time delay

**Step 2: For each point, find nearest neighbor**

For point $\mathbf{v}_i$, find $\mathbf{v}_j$ that minimizes $||\mathbf{v}_i - \mathbf{v}_j||$

(But $j$ shouldn't be temporally close to $i$)

**Step 3: Track how distance grows**

Initial distance: $d_0 = ||\mathbf{v}_i - \mathbf{v}_j||$

After k steps: $d_k = ||\mathbf{v}_{i+k} - \mathbf{v}_{j+k}||$

**Step 4: Compute divergence rate**

$$\lambda \approx \frac{1}{k} \ln \frac{d_k}{d_0}$$

**Step 5: Average over many starting points**

### How to Verify

```python
import numpy as np
from scipy.spatial.distance import cdist

def lyapunov_simple(x, m=3, tau=1, k_max=10):
    """Simplified Lyapunov estimation"""
    n = len(x)
    n_vectors = n - (m-1)*tau

    # Create embedded vectors
    embedded = np.zeros((n_vectors, m))
    for i in range(n_vectors):
        for j in range(m):
            embedded[i, j] = x[i + j*tau]

    # Find nearest neighbors and track divergence
    distances = cdist(embedded, embedded)
    divergences = []

    for i in range(n_vectors - k_max):
        # Exclude temporal neighbors
        dist_row = distances[i].copy()
        dist_row[max(0,i-3):min(n_vectors,i+4)] = np.inf

        j = np.argmin(dist_row)
        d0 = dist_row[j]

        if d0 < 1e-10:
            continue

        # Track divergence
        for k in range(1, k_max):
            if i+k < n_vectors and j+k < n_vectors:
                dk = np.linalg.norm(embedded[i+k] - embedded[j+k])
                if dk > 1e-10:
                    divergences.append((k, np.log(dk/d0)))

    # Linear regression
    if len(divergences) < 10:
        return 0.0

    times = np.array([d[0] for d in divergences])
    log_divs = np.array([d[1] for d in divergences])

    slope, _ = np.polyfit(times, log_divs, 1)
    return slope
```

### Interpretation

| λ Value | Meaning |
|---------|---------|
| λ < 0 | Stable attractor (converging) |
| λ ≈ 0 | Edge of chaos |
| λ > 0 | Chaotic (sensitive to initial conditions) |

---

## 1.9 Recurrence Quantification Analysis (RQA)

**File:** `recurrence/rqa.py`
**Windowed:** YES
**What it measures:** Recurrence patterns in phase space.

### Formula

**Recurrence Matrix:**
$$R_{ij} = \Theta(\epsilon - ||\mathbf{v}_i - \mathbf{v}_j||)$$

Where $\Theta$ is the Heaviside function (1 if true, 0 if false).

**Recurrence Rate:**
$$RR = \frac{1}{N^2} \sum_{i,j=1}^{N} R_{ij}$$

**Determinism:**
$$DET = \frac{\sum_{l=l_{min}}^{N} l \cdot P(l)}{\sum_{l=1}^{N} l \cdot P(l)}$$

Where $P(l)$ = histogram of diagonal line lengths

### Step-by-Step Derivation

**Given:** Time series embedded as vectors $\mathbf{v}_1, ..., \mathbf{v}_N$

**Step 1: Choose threshold ε**

Typically: $\epsilon = 0.1 \times \text{std}(\text{distances})$

**Step 2: Build recurrence matrix**

```
R = [
  [1, 0, 1, 0, 1],
  [0, 1, 0, 1, 0],
  [1, 0, 1, 0, 1],
  [0, 1, 0, 1, 0],
  [1, 0, 1, 0, 1]
]
```
(1 = points are within ε of each other)

**Step 3: Compute Recurrence Rate**
$$RR = \frac{\text{count of 1s}}{N^2} = \frac{13}{25} = 0.52$$

**Step 4: Find diagonal lines**

Diagonal lines indicate deterministic structure.

**Step 5: Compute Determinism**

Count lengths of diagonal lines (excluding main diagonal):
- Length 1: 4 occurrences
- Length 2: 2 occurrences
- Length 3: 0 occurrences

$$DET = \frac{2 \times 2}{1 \times 4 + 2 \times 2} = \frac{4}{8} = 0.5$$

### How to Verify

```python
import numpy as np
from scipy.spatial.distance import cdist

def rqa_simple(x, m=2, tau=1, epsilon=0.1):
    """Simplified RQA"""
    n = len(x)
    n_vec = n - (m-1)*tau

    # Embed
    embedded = np.zeros((n_vec, m))
    for i in range(n_vec):
        for j in range(m):
            embedded[i, j] = x[i + j*tau]

    # Distance matrix
    D = cdist(embedded, embedded)

    # Recurrence matrix
    eps = epsilon * np.std(D)
    R = (D < eps).astype(int)

    # Recurrence rate
    RR = np.sum(R) / (n_vec * n_vec)

    return {'recurrence_rate': RR, 'R': R}
```

### Interpretation

| Metric | High Value Means |
|--------|------------------|
| RR | Many recurrences (periodic or noisy) |
| DET | Deterministic dynamics |
| LAM | Laminar states (system gets "stuck") |

---

## 1.10 Spectral Features

**File:** `frequency/spectral.py`
**Windowed:** NO
**What it measures:** Frequency domain characteristics.

### Formulas

**FFT:**
$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N}$$

**Power Spectrum:**
$$P_k = |X_k|^2$$

**Spectral Centroid:**
$$C = \frac{\sum_k f_k \cdot P_k}{\sum_k P_k}$$

**Spectral Bandwidth:**
$$B = \sqrt{\frac{\sum_k (f_k - C)^2 \cdot P_k}{\sum_k P_k}}$$

### Step-by-Step Derivation

**Given:** Signal $x = [1, 2, 3, 2, 1, 0, 1, 2]$

**Step 1: Remove mean**
$$x_{centered} = x - \bar{x} = [-0.5, 0.5, 1.5, 0.5, -0.5, -1.5, -0.5, 0.5]$$

**Step 2: Compute FFT**

Use numpy.fft.fft() - this gives complex numbers.

**Step 3: Compute power spectrum**
$$P = |FFT|^2$$

Only use positive frequencies (first half).

**Step 4: Get frequency bins**
$$f_k = \frac{k}{N} \quad \text{for } k = 0, 1, ..., N/2$$

**Step 5: Compute centroid**
$$C = \frac{f_1 P_1 + f_2 P_2 + ...}{P_1 + P_2 + ...}$$

### How to Verify

```python
import numpy as np

def spectral_features(x):
    x = x - np.mean(x)
    n = len(x)

    fft_vals = np.fft.fft(x)
    power = np.abs(fft_vals[:n//2])**2
    freqs = np.fft.fftfreq(n)[:n//2]

    # Skip DC component
    power = power[1:]
    freqs = freqs[1:]

    # Normalize
    power = power / np.sum(power)

    # Centroid
    centroid = np.sum(freqs * power)

    # Bandwidth
    bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * power))

    return {'centroid': centroid, 'bandwidth': bandwidth}
```

### Interpretation

| Feature | High Value Means |
|---------|------------------|
| Centroid | Higher dominant frequency |
| Bandwidth | Energy spread across frequencies (noisy) |
| Low bandwidth | Narrowband (periodic) |

---

# Stage 2: Structural Geometry

These engines analyze **multiple signals** to find pairwise relationships.

---

## 2.1 PCA (Principal Component Analysis)

**File:** `geometry/pca.py`
**Windowed:** NO
**What it measures:** Dimensionality reduction. How many independent directions?

### Formula

Find eigenvectors of covariance matrix:
$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

Where:
- $\Sigma$ = covariance matrix
- $\mathbf{v}$ = eigenvector (principal component direction)
- $\lambda$ = eigenvalue (variance explained)

### Step-by-Step Derivation

**Given:** Data matrix X (n samples × p features)

**Step 1: Center the data**
$$X_{centered} = X - \bar{X}$$

**Step 2: Compute covariance matrix**
$$\Sigma = \frac{1}{n-1} X_{centered}^T X_{centered}$$

For 2D:
$$\Sigma = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) \\ \text{Cov}(x_1, x_2) & \text{Var}(x_2) \end{bmatrix}$$

**Step 3: Find eigenvalues and eigenvectors**

Solve: $\det(\Sigma - \lambda I) = 0$

**Step 4: Sort by eigenvalue (descending)**

**Step 5: Compute explained variance ratio**
$$\text{EVR}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

### How to Verify

```python
import numpy as np
from sklearn.decomposition import PCA

# Generate correlated data
np.random.seed(42)
x1 = np.random.randn(100)
x2 = 0.8 * x1 + 0.2 * np.random.randn(100)
X = np.column_stack([x1, x2])

# Manual PCA
X_centered = X - np.mean(X, axis=0)
cov = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Sort
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
explained_variance = eigenvalues / np.sum(eigenvalues)

print(f"Explained variance: {explained_variance}")
# Should be ~[0.9, 0.1] for highly correlated data
```

### Interpretation

| PC1 Variance | Meaning |
|--------------|---------|
| ~100% | All signals move together (1D system) |
| ~50% | Independent directions |
| Spread out | High-dimensional dynamics |

---

## 2.2 Mutual Information

**File:** `geometry/mutual_information.py`
**Windowed:** NO
**What it measures:** Nonlinear dependence (in bits).

### Formula

$$I(X; Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}$$

Or equivalently:
$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

### Step-by-Step Derivation

**Given:** Two signals X and Y

**Step 1: Discretize into bins**

X = [0.1, 0.3, 0.8, 0.2, 0.9] → bins [1, 1, 2, 1, 2]
Y = [0.2, 0.4, 0.7, 0.3, 0.8] → bins [1, 1, 2, 1, 2]

**Step 2: Count joint frequencies**

| | Y=1 | Y=2 |
|---|---|---|
| X=1 | 3 | 0 |
| X=2 | 0 | 2 |

**Step 3: Compute joint probabilities**
$$p(X=1, Y=1) = 3/5 = 0.6$$

**Step 4: Compute marginals**
$$p(X=1) = 3/5, \quad p(Y=1) = 3/5$$

**Step 5: Compute MI**
$$I = 0.6 \log_2\frac{0.6}{0.6 \times 0.6} + 0.4 \log_2\frac{0.4}{0.4 \times 0.4}$$
$$I = 0.6 \log_2(1.67) + 0.4 \log_2(2.5) = 0.6(0.74) + 0.4(1.32) = 0.97 \text{ bits}$$

### Approximation via Correlation

For Gaussian data:
$$I(X; Y) \approx -\frac{1}{2} \log_2(1 - \rho^2)$$

Where $\rho$ = Pearson correlation.

### How to Verify

```python
import numpy as np
from scipy.stats import spearmanr

def mi_approx(x, y):
    """MI approximation via Spearman correlation"""
    rho, _ = spearmanr(x, y)
    if abs(rho) > 0.999:
        return np.inf
    return -0.5 * np.log2(1 - rho**2)

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
print(f"MI ≈ {mi_approx(x, y)} bits")  # High (perfect correlation)

y_indep = np.array([3, 1, 4, 2, 5])
print(f"MI ≈ {mi_approx(x, y_indep)} bits")  # Low (less correlated)
```

### Interpretation

| MI | Meaning |
|----|---------|
| 0 | Independent |
| 1 bit | Knowing X gives 1 bit of info about Y |
| High | Strong nonlinear dependence |

---

## 2.3 Clustering (Silhouette Score)

**File:** `geometry/clustering.py`
**Windowed:** NO
**What it measures:** How well-separated are natural clusters?

### Formula

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean distance to points in same cluster
- $b(i)$ = mean distance to points in nearest other cluster

### Step-by-Step Derivation

**Given:** Points assigned to clusters

**Step 1: For point i, compute a(i)**

Point i is in cluster A with points {j, k, l}.
$$a(i) = \frac{d(i,j) + d(i,k) + d(i,l)}{3}$$

**Step 2: For point i, compute b(i)**

Find the nearest cluster B. Compute mean distance to all points in B.

**Step 3: Compute silhouette**
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Step 4: Average over all points**
$$\bar{s} = \frac{1}{n} \sum_i s(i)$$

### How to Verify

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create clustered data
np.random.seed(42)
cluster1 = np.random.randn(50, 2) + [0, 0]
cluster2 = np.random.randn(50, 2) + [5, 5]
X = np.vstack([cluster1, cluster2])

# Cluster
km = KMeans(n_clusters=2, random_state=42)
labels = km.fit_predict(X)

# Silhouette
score = silhouette_score(X, labels)
print(f"Silhouette = {score}")  # High (~0.7) for well-separated clusters
```

### Interpretation

| Score | Meaning |
|-------|---------|
| -1 to 0 | Wrong clustering |
| 0 to 0.5 | Overlapping clusters |
| 0.5 to 1 | Well-separated clusters |

---

# Stage 3: Dynamical Systems

These engines track **how geometry evolves** over time.

---

## 3.1 HD-Slope (Hausdorff Distance Slope)

**File:** `state/trajectory.py`
**Windowed:** NO
**What it measures:** Rate of coherence loss. **THE key prognostic.**

### Formula

$$hd\_slope = \frac{d(\text{distance from baseline})}{dt}$$

Computed via linear regression:
$$d(t) = hd\_slope \cdot t + c$$

### Step-by-Step Derivation

**Given:** Feature vectors $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n$ at times $t_1, t_2, ..., t_n$

**Step 1: Define baseline**
$$\mathbf{v}_{baseline} = \mathbf{v}_1 \quad \text{(first observation)}$$

**Step 2: Compute distance at each time**
$$d(t_i) = ||\mathbf{v}_i - \mathbf{v}_{baseline}||_2$$

**Step 3: Linear regression**

Fit: $d = m \cdot t + b$

$$m = \frac{n\sum t_i d_i - \sum t_i \sum d_i}{n\sum t_i^2 - (\sum t_i)^2}$$

**Step 4: hd_slope = m**

### How to Verify

```python
import numpy as np
from scipy.stats import linregress

# Simulate feature vectors drifting from baseline
np.random.seed(42)
n = 50
times = np.arange(n)
baseline = np.array([0, 0, 0])  # 3D feature space

# Each timestep, drift a bit
features = []
for t in times:
    v = baseline + 0.1 * t + 0.05 * np.random.randn(3)  # Linear drift + noise
    features.append(v)

features = np.array(features)

# Compute distances from baseline
distances = [np.linalg.norm(f - baseline) for f in features]

# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(times, distances)
print(f"hd_slope = {slope:.4f}")
print(f"R² = {r_value**2:.4f}")  # How linear is the drift?
```

### Interpretation

| hd_slope | Meaning |
|----------|---------|
| ~0 | System stable, staying near baseline |
| Positive | System drifting away (degrading) |
| Large positive | Rapid degradation, failure imminent |

**This is the most important metric for prognosis.**

---

## 3.2 Hamiltonian (Total Energy)

**File:** `physics/hamiltonian.py`
**Windowed:** NO
**What it measures:** Energy conservation. Is the system closed?

### Formula

$$H = T + V$$

Where:
- $T = \frac{1}{2}\dot{x}^2$ = kinetic energy
- $V = \frac{1}{2}(x - \bar{x})^2$ = potential energy (deviation from equilibrium)

### Step-by-Step Derivation

**Given:** Time series $x_1, x_2, ..., x_n$

**Step 1: Compute equilibrium**
$$\bar{x} = \text{mean}(x)$$

**Step 2: Compute velocity (first derivative)**
$$\dot{x}_t = x_{t+1} - x_t$$

**Step 3: Compute kinetic energy**
$$T_t = \frac{1}{2} \dot{x}_t^2$$

**Step 4: Compute potential energy**
$$V_t = \frac{1}{2} (x_t - \bar{x})^2$$

**Step 5: Compute Hamiltonian**
$$H_t = T_t + V_t$$

**Step 6: Check conservation**

If $dH/dt \approx 0$, system is conservative.
If $dH/dt > 0$, energy is being injected.
If $dH/dt < 0$, energy is dissipating.

### How to Verify

```python
import numpy as np

def hamiltonian(x):
    x_bar = np.mean(x)

    # Velocity
    dx = np.diff(x)

    # Kinetic energy
    T = 0.5 * dx**2

    # Potential energy (aligned with velocity)
    V = 0.5 * (x[:-1] - x_bar)**2

    # Hamiltonian
    H = T + V

    # Conservation check
    H_trend = np.polyfit(np.arange(len(H)), H, 1)[0]

    return {
        'H': H,
        'H_mean': np.mean(H),
        'H_trend': H_trend,  # dH/dt
        'is_conservative': abs(H_trend) < 0.01 * np.mean(H)
    }

# Test: harmonic oscillator (should be conservative)
t = np.linspace(0, 10, 1000)
x = np.sin(t)  # Perfect sine wave
result = hamiltonian(x)
print(f"H trend = {result['H_trend']:.6f}")  # Should be ~0
```

### Interpretation

| dH/dt | Meaning |
|-------|---------|
| ~0 | Closed system, energy conserved |
| > 0 | External energy input |
| < 0 | Energy dissipation (friction, damping) |

---

# Stage 4: Causal Mechanics

These engines determine **causal relationships** between signals.

---

## 4.1 Granger Causality

**File:** `state/granger.py`
**Windowed:** NO
**What it measures:** Does X help predict Y beyond Y's own history?

### Formula

**Restricted model (Y predicts itself):**
$$Y_t = \sum_{i=1}^{p} \alpha_i Y_{t-i} + \epsilon_t$$

**Unrestricted model (Y + X predict Y):**
$$Y_t = \sum_{i=1}^{p} \alpha_i Y_{t-i} + \sum_{j=1}^{p} \beta_j X_{t-j} + \eta_t$$

**F-test:**
$$F = \frac{(RSS_r - RSS_u) / p}{RSS_u / (n - 2p - 1)}$$

If F is large (p-value small), X Granger-causes Y.

### Step-by-Step Derivation

**Given:** Two time series X and Y

**Step 1: Choose lag p** (e.g., p=2)

**Step 2: Build restricted model**

Regress Y on its own lags:
$$Y_t = \alpha_1 Y_{t-1} + \alpha_2 Y_{t-2} + \epsilon$$

Compute $RSS_r = \sum \epsilon^2$

**Step 3: Build unrestricted model**

Regress Y on its own lags AND X's lags:
$$Y_t = \alpha_1 Y_{t-1} + \alpha_2 Y_{t-2} + \beta_1 X_{t-1} + \beta_2 X_{t-2} + \eta$$

Compute $RSS_u = \sum \eta^2$

**Step 4: F-test**

$$F = \frac{(RSS_r - RSS_u) / 2}{RSS_u / (n - 5)}$$

**Step 5: Get p-value from F-distribution**

### How to Verify

```python
import numpy as np
from scipy import stats

def granger_causality(x, y, lag=2):
    """Test if X Granger-causes Y"""
    n = len(y) - lag

    # Build lagged matrices
    Y = y[lag:]  # Target

    # Restricted: only Y lags
    Y_lags = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])

    # Unrestricted: Y lags + X lags
    X_lags = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])
    Z = np.column_stack([Y_lags, X_lags])

    # Fit models
    # Restricted
    beta_r = np.linalg.lstsq(Y_lags, Y, rcond=None)[0]
    resid_r = Y - Y_lags @ beta_r
    RSS_r = np.sum(resid_r**2)

    # Unrestricted
    beta_u = np.linalg.lstsq(Z, Y, rcond=None)[0]
    resid_u = Y - Z @ beta_u
    RSS_u = np.sum(resid_u**2)

    # F-test
    df1 = lag  # Additional parameters
    df2 = n - 2*lag - 1
    F = ((RSS_r - RSS_u) / df1) / (RSS_u / df2)
    p_value = 1 - stats.f.cdf(F, df1, df2)

    return {'F': F, 'p_value': p_value, 'X_causes_Y': p_value < 0.05}

# Test: X causes Y
np.random.seed(42)
x = np.random.randn(100)
y = np.zeros(100)
for t in range(2, 100):
    y[t] = 0.5 * y[t-1] + 0.3 * x[t-1] + 0.1 * np.random.randn()

result = granger_causality(x, y)
print(f"F = {result['F']:.2f}, p = {result['p_value']:.4f}")
print(f"X causes Y: {result['X_causes_Y']}")
```

### Interpretation

| p-value | Meaning |
|---------|---------|
| < 0.05 | X Granger-causes Y |
| > 0.05 | No evidence X predicts Y |

**Note:** Granger causality is not true causality - it's predictive causality.

---

## 4.2 Transfer Entropy

**File:** `state/transfer_entropy.py`
**Windowed:** NO
**What it measures:** Information flow (in bits) from X to Y.

### Formula

$$TE_{X \to Y} = \sum p(y_{t+1}, y_t, x_t) \log_2 \frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}$$

Or equivalently:
$$TE_{X \to Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)$$

### Step-by-Step Derivation

**Given:** Time series X and Y

**Step 1: Discretize both series into bins**

X → [1, 2, 1, 2, 3, 2, 1, ...]
Y → [2, 2, 3, 3, 2, 1, 2, ...]

**Step 2: Create joint states**

For each t, record: $(y_{t+1}, y_t, x_t)$

**Step 3: Count frequencies**

Build 3D histogram of $(y_{t+1}, y_t, x_t)$ combinations.

**Step 4: Compute conditional probabilities**

$$p(y_{t+1} | y_t, x_t) = \frac{\text{count}(y_{t+1}, y_t, x_t)}{\text{count}(y_t, x_t)}$$

$$p(y_{t+1} | y_t) = \frac{\text{count}(y_{t+1}, y_t)}{\text{count}(y_t)}$$

**Step 5: Sum over all states**

$$TE = \sum_{y_{t+1}, y_t, x_t} p(y_{t+1}, y_t, x_t) \log_2 \frac{p(y_{t+1} | y_t, x_t)}{p(y_{t+1} | y_t)}$$

### How to Verify

```python
import numpy as np
from collections import Counter

def transfer_entropy(x, y, bins=5):
    """Compute TE from X to Y"""
    n = len(x) - 1

    # Discretize
    x_d = np.digitize(x, np.linspace(min(x), max(x), bins))
    y_d = np.digitize(y, np.linspace(min(y), max(y), bins))

    # Create states: (y_{t+1}, y_t, x_t)
    states_3 = [(y_d[t+1], y_d[t], x_d[t]) for t in range(n)]
    states_2_yx = [(y_d[t+1], y_d[t]) for t in range(n)]
    states_2_xy = [(y_d[t], x_d[t]) for t in range(n)]
    states_1_y = [y_d[t] for t in range(n)]

    # Count
    count_3 = Counter(states_3)
    count_2_yx = Counter(states_2_yx)
    count_2_xy = Counter(states_2_xy)
    count_1_y = Counter(states_1_y)

    # Compute TE
    TE = 0
    for (y_next, y_curr, x_curr), count in count_3.items():
        p_joint = count / n
        p_y_next_given_y_x = count / count_2_xy[(y_curr, x_curr)]
        p_y_next_given_y = count_2_yx[(y_next, y_curr)] / count_1_y[y_curr]

        if p_y_next_given_y > 0:
            TE += p_joint * np.log2(p_y_next_given_y_x / p_y_next_given_y)

    return TE

# Test: X drives Y
np.random.seed(42)
x = np.cumsum(np.random.randn(200))
y = np.zeros(200)
for t in range(1, 200):
    y[t] = 0.5 * y[t-1] + 0.5 * x[t-1] + 0.1 * np.random.randn()

te_x_to_y = transfer_entropy(x, y)
te_y_to_x = transfer_entropy(y, x)
print(f"TE(X→Y) = {te_x_to_y:.4f} bits")
print(f"TE(Y→X) = {te_y_to_x:.4f} bits")
print(f"Net flow: {'X→Y' if te_x_to_y > te_y_to_x else 'Y→X'}")
```

### Interpretation

| TE | Meaning |
|----|---------|
| 0 | No information flow |
| Positive | X provides info about Y's future |
| TE(X→Y) > TE(Y→X) | Net flow from X to Y |

---

# Engine Interface

All engines follow this pattern:

```python
def compute(
    series: np.ndarray,
    mode: str = 'static',      # 'static', 'windowed', or 'point'
    t: Optional[int] = None,   # Required for 'point' mode
    window_size: int = 200,    # For windowed/point modes
    step_size: int = 20,       # For windowed mode
    **kwargs                   # Engine-specific params
) -> Dict[str, Any]:
    """
    Returns:
        mode='static': Single values (testing only)
        mode='windowed': Arrays with 't' index
        mode='point': Single values at time t
    """
```

### Mode Summary

| Mode | Use Case | Output |
|------|----------|--------|
| `static` | Testing only | Single value over entire signal |
| `windowed` | Pre-compute | Array of values + timestamps |
| `point` | Query | Single value at specific time t |

**Philosophy:** `windowed` engines compute once, store results, then queries are O(1) lookups.

---

# Quick Reference Card

| Engine | Formula | Interpretation |
|--------|---------|----------------|
| Hurst | $\log(R/S) \propto H \log(n)$ | H>0.5 trending, H<0.5 reverting |
| PermEnt | $-\sum p_i \log p_i$ | High = complex |
| SampEn | $-\ln(A/B)$ | Low = regular |
| GARCH | $\sigma^2 = \omega + \alpha\epsilon^2 + \beta\sigma^2$ | Volatility clustering |
| RealVol | $\sqrt{\sum r^2}$ | Actual observed vol |
| Lyapunov | $\lambda = \lim \frac{1}{t}\ln\frac{d(t)}{d(0)}$ | λ>0 = chaos |
| RQA | $RR = \frac{\sum R_{ij}}{N^2}$ | Recurrence patterns |
| PCA | $\Sigma v = \lambda v$ | Dimensionality |
| MI | $H(X) + H(Y) - H(X,Y)$ | Nonlinear dependence |
| hd_slope | $\frac{d(\|v-v_0\|)}{dt}$ | Degradation rate |
| Granger | F-test on $RSS_r$ vs $RSS_u$ | Predictive causality |
| TransEnt | $H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t)$ | Information flow |

---

*Last updated: 2025-01-23*
*For the 2am grad student: if the code doesn't match these formulas, the code is wrong.*
