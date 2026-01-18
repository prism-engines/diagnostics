# PRISM Validation Suite

This directory contains validation studies testing PRISM against systems with known ground truth.

## Philosophy

PRISM validation follows the scientific method:

1. **Known physics** → Generate synthetic data with analytically solvable equations
2. **Ground truth labels** → Tag each trajectory with its known regime/behavior
3. **Blind analysis** → Run PRISM without using the labels
4. **Compare** → Do PRISM metrics correlate with ground truth?

## Validation Studies

| Study | System | Ground Truth | Data Type | Status |
|-------|--------|--------------|-----------|--------|
| [01_double_pendulum.md](01_double_pendulum.md) | Nonlinear dynamics | Chaos transition at ~60° | Synthetic | ✓ Complete |
| [02_chemical_kinetics.md](02_chemical_kinetics.md) | Reaction kinetics | Rate laws, Arrhenius | Synthetic | ✓ Complete |
| [03_chemked_combustion.md](03_chemked_combustion.md) | Combustion kinetics | Arrhenius, ignition delay | **Real experimental** | ✓ Complete |
| [04_gray_scott_pde.md](04_gray_scott_pde.md) | Reaction-diffusion | 6 pattern regimes | **PDE simulation** | ✓ Complete |
| [05_sabiork_enzyme_kinetics.md](05_sabiork_enzyme_kinetics.md) | Enzyme kinetics | Michaelis-Menten regimes | **API + Simulation** | ✓ Complete |
| [06_physionet_ecg.md](06_physionet_ecg.md) | Cardiac rhythms | Beat annotations | **Real clinical** | ✓ Complete |
| [07_mimic_sepsis.md](07_mimic_sepsis.md) | ICU sepsis | Diagnosis codes | **Real clinical** | ✓ Complete (Association) |

## Results Summary

### Double Pendulum

| Test | Result | Key Metric |
|------|--------|------------|
| Chaos detection | **PASS** | Sample Entropy +75% |
| Energy conservation | **PASS** | CV < 10⁻⁶ |
| Lyapunov correlation | **PASS** | Positive with chaos |

### Chemical Kinetics (Synthetic)

| Test | Result | Key Metric |
|------|--------|------------|
| Reaction order discrimination | **PASS** | Hurst 0.53 vs 0.95 |
| Oscillation detection | **PASS** | Lyapunov ≈ 0 for limit cycles |
| Rate constant recovery | **Expected limitation** | k is scale, not shape |

### ChemKED Combustion (Real Data)

| Test | Result | Key Metric |
|------|--------|------------|
| Arrhenius fit correlation | **PASS** | Hurst +0.54 with R² |
| Data quality detection | **PASS** | Entropy -0.55 with R² |
| Fuel fingerprinting | **Exploratory** | Distinct Hurst by fuel |

### Gray-Scott PDE (Simulation)

| Test | Result | Key Metric |
|------|--------|------------|
| Regime discrimination | **PASS** | ANOVA F=74.5, p<0.0001 |
| Worms identification | **PASS** | Unique: H=0.64, SampEn=1.38 |
| Stationary vs dynamic | **PASS** | Clear entropy separation |

### SABIO-RK Enzyme Kinetics (API Data)

| Test | Result | Key Metric |
|------|--------|------------|
| Regime discrimination | **PASS** | ANOVA F=236.4, p<0.0001 |
| Linear vs saturating | **PASS** | SampEn 1.48 vs 0.37 |
| Physical interpretation | **PASS** | Entropy correlates with kinetic order |

### PhysioNet ECG (Clinical Data)

| Test | Result | Key Metric |
|------|--------|------------|
| Regime discrimination | **PASS** | ANOVA F=13.3, p<0.0001 |
| Normal vs arrhythmia | **PASS** | SampEn 0.58 vs 0.43 |
| Clinical interpretation | **PASS** | Lower entropy = loss of HRV |

### MIMIC-IV Sepsis (Clinical Data) - Association Study

| Test | Result | Key Metric |
|------|--------|------------|
| Vector: Regime discrimination | **PASS** | ANOVA F=65.8, p<0.0001 |
| Vector: Stable vs septic | **PASS** | SampEn 1.32 vs 0.93 |
| **Geometry: Correlation** | **PASS** | |Pearson| 0.24 vs 0.27, F=14.77, p=0.0001 |
| Geometry: Transfer Entropy | Trend | TE 1.86 vs 1.98, p=0.16 |
| Geometry: Cointegration | No effect | Both ~100% cointegrated |
| **Coupling Trajectory** | **NOT TESTABLE** | Only 1 late-onset patient in demo |
| Early warning | **NOT TESTED** | Dataset structurally incapable |

**Critical Note:** The demo dataset cannot answer the early warning question because 91% of patients had infection at/before ICU admission. The correct research question is whether the **rate of change** in vital sign coupling (the derivative) predicts sepsis - not absolute values. This requires late-onset cases with pre-sepsis vital trajectories. See [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/) for proper methodology.

## Regenerating Validation Data

All validation data can be regenerated from first principles:

```bash
# Double pendulum (synthetic)
python scripts/double_pendulum.py
python scripts/validate_double_pendulum.py

# Chemical kinetics (synthetic)
python scripts/chemical_kinetics.py
python scripts/validate_chemical_kinetics.py

# ChemKED combustion (real experimental data)
cd data && git clone https://github.com/pr-omethe-us/ChemKED-database.git chemked
python fetchers/chemked_fetcher.py
python scripts/validate_chemked.py

# Gray-Scott reaction-diffusion (PDE simulation from The Well)
pip install the-well
python fetchers/the_well_fetcher.py 12
python scripts/validate_gray_scott.py

# SABIO-RK enzyme kinetics (API data)
python fetchers/sabiork_fetcher.py --max-entries 100
python scripts/validate_sabiork.py

# PhysioNet MIT-BIH ECG (clinical data)
pip install wfdb antropy
python fetchers/physionet_fetcher.py --records 20
python scripts/validate_physionet.py

# MIMIC-IV ICU Sepsis (clinical data)
pip install requests antropy
python fetchers/mimic_fetcher.py --demo
python scripts/validate_mimic.py
```

## Future Validation Studies

### Planned

| System | Physics | PRISM Test |
|--------|---------|------------|
| Lorenz attractor | Chaos, strange attractors | RQA, Lyapunov |
| Logistic map | Period doubling | Bifurcation detection |
| Van der Pol oscillator | Relaxation oscillations | Limit cycle geometry |
| Ising model | Phase transitions | Critical exponents |

### Experimental & Simulation Data Sources

| Database | Domain | URL |
|----------|--------|-----|
| NIST Chemical Kinetics | Gas-phase reactions | https://kinetics.nist.gov/ |
| ReSpecTh | Combustion kinetics | https://respecth.chem.elte.hu/ |
| **The Well** | **PDE simulations (15TB)** | https://github.com/PolymathicAI/the_well |
| **SABIO-RK** | **Enzyme kinetics (71K+ entries)** | https://sabiork.h-its.org/ |
| UCI Machine Learning | Benchmark datasets | https://archive.ics.uci.edu/ |
| PhysioNet | Physiological signals | https://physionet.org/ |

## Academic Standards

All validation follows academic research standards:

- **NO SHORTCUTS** - Full algorithms, no subsampling
- **NO APPROXIMATIONS** - Peer-reviewed implementations
- **REPRODUCIBLE** - Scripts generate data from equations
- **REFERENCED** - All methods cite original papers

## Core References

### Nonlinear Dynamics

1. Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.
2. Ott, E. (2002). *Chaos in Dynamical Systems* (2nd ed.). Cambridge University Press.
3. Kantz, H., & Schreiber, T. (2004). *Nonlinear Signal Topology Analysis* (2nd ed.). Cambridge University Press.

### Signal Topology Analysis

4. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Signal Topology Analysis: Forecasting and Control* (5th ed.). Wiley.
5. Hamilton, J. D. (1994). *Signal Topology Analysis*. Princeton University Press.

### Information Theory

6. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
7. Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

---


