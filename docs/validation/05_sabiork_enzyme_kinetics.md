# PRISM Validation: SABIO-RK Enzyme Kinetics

## Overview

This document validates PRISM against enzyme kinetics data from the SABIO-RK database. The Michaelis-Menten model provides ground truth for testing whether PRISM can distinguish substrate saturation regimes.

## Data Source

**SABIO-RK Database**
- Repository: https://sabiork.h-its.org/
- Paper: Wittig et al. (2012), Nucleic Acids Research
- License: Free for academic use
- Total entries: 71,000+ kinetic laws

**Michaelis-Menten Subset:**
- Entries: 31,649 with Michaelis-Menten kinetics
- Fetched: 50 entries (11 with valid Km and Vmax)
- Indicators: 33 (3 regimes per entry)
- Observations: 3,300 (100 points per trajectory)

## Reproducibility

```bash
# Install PRISM
pip install -e .

# Fetch SABIO-RK data
python fetchers/sabiork_fetcher.py --max-entries 100

# Run validation
python scripts/validate_sabiork.py
```

---

## 1. Michaelis-Menten Kinetics

### Rate Equation

The Michaelis-Menten model describes enzyme-catalyzed reactions:

```
v = Vmax × [S] / (Km + [S])
```

Where:
- v = reaction velocity
- Vmax = maximum velocity (at substrate saturation)
- [S] = substrate concentration
- Km = Michaelis constant (substrate concentration at half-max velocity)

### Kinetic Regimes

Different substrate concentrations produce distinct dynamical regimes:

| Regime | Condition | Behavior |
|--------|-----------|----------|
| Linear | [S] << Km | v ≈ (Vmax/Km) × [S] (first-order) |
| Transition | [S] ≈ Km | v = Vmax/2 (half-saturation) |
| Saturating | [S] >> Km | v ≈ Vmax (zero-order plateau) |

---

## 2. PRISM Test Design

### Hypothesis

Different kinetic regimes should have characteristic entropy signatures:

- **Linear regime**: Variable velocity → Higher entropy
- **Transition regime**: Intermediate behavior → Moderate entropy
- **Saturating regime**: Plateau dynamics → Lower entropy

### Data Simulation

For each SABIO-RK entry with valid Km and Vmax, we simulated three trajectories:

1. **Linear**: [S] in range [0.01×Km, 0.2×Km]
2. **Transition**: [S] in range [0.5×Km, 2×Km]
3. **Saturating**: [S] in range [5×Km, 20×Km]

Each trajectory contains 100 velocity measurements with 2% noise.

---

## 3. Validation Results

### Test: Regime Discrimination (ANOVA)

**Table 1: Mean PRISM metrics by kinetic regime**

| Regime | Sample Entropy | Permutation Entropy | n |
|--------|----------------|---------------------|---|
| Linear | 1.48 | 0.99 | 11 |
| Transition | 0.45 | 0.95 | 11 |
| Saturating | 0.37 | 0.97 | 11 |

**Statistical Tests:**

| Metric | F-statistic | p-value | Significant |
|--------|-------------|---------|-------------|
| Sample Entropy | 236.44 | < 0.0001 | **Yes*** |
| Permutation Entropy | 2.86 | 0.074 | No |

### Key Findings

1. **Sample entropy strongly discriminates regimes**: F = 236.44, p < 0.0001

2. **Physical interpretation verified**:
   - Linear regime: High entropy (1.48) — variable velocity
   - Saturating regime: Low entropy (0.37) — plateau behavior

3. **Clear monotonic trend**: Linear > Transition > Saturating

---

## 4. Interpretation

### Physical Meaning of PRISM Metrics

| Metric | Physical Interpretation for Enzyme Kinetics |
|--------|---------------------------------------------|
| High Sample Entropy | Variable dynamics (first-order kinetics) |
| Low Sample Entropy | Plateau dynamics (zero-order kinetics) |
| Permutation Entropy | Ordinal complexity (less discriminative) |

### Enzyme-Specific Results

| Enzyme | Km | Regime | Sample Entropy |
|--------|-----|--------|----------------|
| NADPH | 19 μM | Linear | 1.33 |
| NADPH | 19 μM | Saturating | 0.39 |
| Na+/K+-ATPase | 6.5 mM | Linear | 1.45 |
| Na+/K+-ATPase | 6.5 mM | Saturating | 0.40 |

The entropy drop from linear to saturating is consistent across enzymes.

---

## 5. Conclusions

PRISM successfully distinguishes Michaelis-Menten kinetic regimes:

| Test | Result | Evidence |
|------|--------|----------|
| Regime discrimination | **PASS** | ANOVA F=236.44, p<0.0001 |
| Linear vs saturating | **PASS** | SampEn 1.48 vs 0.37 |
| Physical interpretation | **PASS** | Entropy correlates with kinetic order |

**Key Insight:** Sample entropy captures the transition from first-order (linear) to zero-order (saturating) kinetics, even from velocity signal topology alone.

---

## Academic References

### SABIO-RK Database

1. **Wittig, U., Kania, R., Golebiewski, M., et al.** (2012). SABIO-RK—database for biochemical reaction kinetics. *Nucleic Acids Research*, 40(D1), D790-D798.
   - DOI: [10.1093/nar/gkr1046](https://doi.org/10.1093/nar/gkr1046)
   - Database reference

### Michaelis-Menten Kinetics

2. **Michaelis, L., & Menten, M. L.** (1913). Die Kinetik der Invertinwirkung. *Biochemische Zeitschrift*, 49, 333-369.
   - Original paper introducing the model
   - English translation in *FEBS Letters* (2013)

3. **Briggs, G. E., & Haldane, J. B. S.** (1925). A note on the kinetics of enzyme action. *Biochemical Journal*, 19(2), 338-339.
   - DOI: [10.1042/bj0190338](https://doi.org/10.1042/bj0190338)
   - Steady-state derivation

4. **Johnson, K. A., & Goody, R. S.** (2011). The original Michaelis constant: translation of the 1913 Michaelis-Menten paper. *Biochemistry*, 50(39), 8264-8269.
   - DOI: [10.1021/bi201284u](https://doi.org/10.1021/bi201284u)
   - Modern interpretation

### Enzyme Kinetics Analysis

5. **Cornish-Bowden, A.** (2012). *Fundamentals of Enzyme Kinetics* (4th ed.). Wiley-Blackwell.
   - ISBN: 978-3527330744
   - Comprehensive textbook

6. **Fersht, A.** (1999). *Structure and Mechanism in Protein Science*. W.H. Freeman.
   - ISBN: 978-0716732686
   - Enzyme mechanism reference

---

## Data Availability

```
data/sabiork/
├── raw/
│   ├── observations.parquet   # Velocity signal topology
│   └── indicators.parquet     # Enzyme metadata
├── config/
│   ├── cohorts.parquet
│   └── cohort_members.parquet
└── vector/
    └── indicator.parquet      # PRISM metrics
```

### Indicator Schema

| Column | Description |
|--------|-------------|
| indicator_id | sabiork_{entry_id}_{regime} |
| enzyme_name | Enzyme or substrate name |
| km | Michaelis constant |
| vmax | Maximum velocity |
| regime | linear, transition, saturating |
| n_points | Number of observations |
