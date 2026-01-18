"""
PRISM Domain Configurations
===========================

YAML configuration files for different scientific domains.
Each domain declares its temporal characteristics:
- Resolution bounds (min/max)
- Frequency definitions
- Window sizes
- Validation thresholds

Available domains:
- cmapss: NASA turbofan engine degradation
- climate: Climate and environmental signal topology
- archaeology: Archaeological and geological signal topology
- particle_physics: Particle collision and decay measurements
- chemistry: Chemical reaction kinetics and spectroscopy
- eeg: Electroencephalography brain signal analysis
- seismology: Seismic wave analysis

Usage:
    from prism.config.domain import get_domain_config

    config = get_domain_config('cmapss')
    window_days = config.default_window.to_base_units('days')
"""

from pathlib import Path

DOMAINS_DIR = Path(__file__).parent

__all__ = ['DOMAINS_DIR']
