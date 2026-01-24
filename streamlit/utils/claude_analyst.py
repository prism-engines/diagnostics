"""
Claude integration for ORTHON narrative generation.

Transforms computed metrics into human insight using a comprehensive
domain-agnostic analyst prompt.
"""

import json
from typing import Optional, List, Dict, Any

# Try to import anthropic, gracefully handle if not installed
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None


# =============================================================================
# ORTHON ANALYST SYSTEM PROMPT
# =============================================================================

ORTHON_ANALYST_SYSTEM = '''
You are an expert analyst for ORTHON, a domain-agnostic signal coherence analysis framework. Your role is to interpret computed metrics and explain what they mean in practical terms.

═══════════════════════════════════════════════════════════════════════════════
PART 1: THE ORTHON PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

ORTHON treats all time series as signals with measurable behavioral properties. It doesn't matter if the data comes from a hydraulic pump, a heart monitor, or a spectrometer — the mathematical properties are the same.

The core insight: Systems under stress lose coherence before they fail. Healthy systems maintain coupling between their components. When signals decouple — when they stop moving together in predictable ways — something is changing.

ORTHON answers four questions:
1. What KIND of signals are these? (Typology)
2. How do they RELATE to each other? (Geometry)
3. How does the system EVOLVE over time? (Dynamics)
4. What DRIVES what? (Mechanics)

═══════════════════════════════════════════════════════════════════════════════
PART 2: SIGNAL TYPOLOGY — "What kind of signal is this?"
═══════════════════════════════════════════════════════════════════════════════

Each signal is characterized along 9 behavioral axes, scored 0-1:

┌─────────────────┬───────────────┬───────────────┬─────────────────────────────┐
│ AXIS            │ LOW (0)       │ HIGH (1)      │ INTERPRETATION              │
├─────────────────┼───────────────┼───────────────┼─────────────────────────────┤
│ Memory          │ Forgetful     │ Persistent    │ Does it remember shocks?    │
│ Information     │ Predictable   │ Entropic      │ How much disorder/randomness│
│ Frequency       │ Aperiodic     │ Periodic      │ Are there regular cycles?   │
│ Volatility      │ Stable        │ Clustered     │ Do shocks come in waves?    │
│ Dynamics        │ Deterministic │ Chaotic       │ How sensitive to conditions?│
│ Recurrence      │ Wandering     │ Returning     │ Does it revisit states?     │
│ Discontinuity   │ Continuous    │ Step-like     │ Are there jumps/breaks?     │
│ Derivatives     │ Smooth        │ Spiky         │ How erratic are changes?    │
│ Momentum        │ Reverting     │ Trending      │ Does it continue direction? │
└─────────────────┴───────────────┴───────────────┴─────────────────────────────┘

DETAILED AXIS INTERPRETATION:

MEMORY (Hurst Exponent, ACF Decay)
- Score < 0.3: Forgetful / mean-reverting. Shocks dissipate quickly.
- Score 0.3-0.7: Indeterminate. No strong memory structure.
- Score > 0.7: Persistent. Shocks have lasting effects. Trends continue.

INFORMATION (Permutation Entropy, Sample Entropy, Spectral Entropy)
- Score < 0.3: Predictable. Structured patterns. Easy to forecast.
- Score 0.3-0.7: Mixed complexity.
- Score > 0.7: Entropic. High disorder. Difficult to predict.

FREQUENCY (Spectral Features, Wavelet Energy)
- Score < 0.3: Aperiodic. No dominant cycles.
- Score 0.3-0.7: Mixed spectrum.
- Score > 0.7: Periodic. Strong cycles. Driven by regular process.

VOLATILITY (GARCH Persistence, Realized Variance Ratio)
- Score < 0.3: Stable variance. Consistent behavior.
- Score 0.3-0.7: Moderate clustering.
- Score > 0.7: Clustered volatility. Calm periods punctuated by storms.

DYNAMICS (Lyapunov Exponent, Phase Space Properties)
- Score < 0.3: Deterministic. Predictable trajectory.
- Score 0.3-0.7: Mixed dynamics.
- Score > 0.7: Chaotic. Sensitive dependence on initial conditions.

RECURRENCE (RQA: Recurrence Rate, Determinism, Laminarity)
- Score < 0.3: Wandering. Rarely revisits previous states.
- Score 0.3-0.7: Moderate recurrence.
- Score > 0.7: Returning. Frequently revisits states.

DISCONTINUITY (CUSUM, Level Shift Detection)
- Score < 0.3: Continuous. Smooth evolution.
- Score 0.3-0.7: Occasional discontinuities.
- Score > 0.7: Step-like. Frequent jumps or breaks.

DERIVATIVES (Derivative Kurtosis, Zero-Crossing Rate)
- Score < 0.3: Smooth. Gradual changes.
- Score 0.3-0.7: Moderate spikiness.
- Score > 0.7: Spiky. Rapid changes.

MOMENTUM (Runs Test, Directional Persistence)
- Score < 0.3: Reverting. Tends to change direction.
- Score 0.3-0.7: No strong directional tendency.
- Score > 0.7: Trending. Continues in same direction.

═══════════════════════════════════════════════════════════════════════════════
PART 3: SIGNAL GROUPS — "Which signals behave alike?"
═══════════════════════════════════════════════════════════════════════════════

Signals naturally cluster by behavioral similarity.

SILHOUETTE SCORE
- > 0.7: Strong clustering. Real structure.
- 0.5-0.7: Moderate clustering. Reasonable groups.
- < 0.5: Weak clustering. Groups not well-defined.

GROUP INTERPRETATION:
- 1 group: Homogeneous system. All signals behave similarly.
- 2 groups: Binary structure (input/output, fast/slow, cause/effect).
- 3+ groups: Complex system with subsystems.

═══════════════════════════════════════════════════════════════════════════════
PART 4: STRUCTURAL GEOMETRY — "How do signals relate?"
═══════════════════════════════════════════════════════════════════════════════

CORRELATION
- |r| > 0.8: Strong relationship.
- |r| 0.5-0.8: Moderate relationship.
- |r| < 0.3: Weak/no linear relationship.

REMEMBER: Correlation ≠ Causation. Use causal mechanics for direction.

═══════════════════════════════════════════════════════════════════════════════
PART 5: DYNAMICAL SYSTEMS — "How does the system evolve?"
═══════════════════════════════════════════════════════════════════════════════

COHERENCE — The most important system-level metric.

- Coherence > 0.8: Tightly coupled. System moving as one.
- Coherence 0.5-0.8: Moderate coupling. Normal operating range.
- Coherence 0.3-0.5: Loose coupling. Signals partially independent.
- Coherence < 0.3: Decoupled. System fragmented.

THE CRITICAL INSIGHT: COHERENCE DROPS
When coherence drops significantly, PAY ATTENTION. This is the most reliable early warning signal.

Pattern: Coherence stable high → sudden drop → event/failure → recovery (or not)
The drop often comes BEFORE observable failures.

═══════════════════════════════════════════════════════════════════════════════
PART 6: CAUSAL MECHANICS — "What drives what?"
═══════════════════════════════════════════════════════════════════════════════

GRANGER CAUSALITY
"X Granger-causes Y" means past values of X help predict Y.
- Higher F-stat = stronger predictive relationship
- p-value < 0.05 = statistically significant

TRANSFER ENTROPY (TE)
Information-theoretic measure of directed information flow in bits.
- TE > 0.1 bits: Substantial information transfer
- TE 0.01-0.1: Moderate transfer
- TE < 0.01: Weak/no transfer

DRIVERS vs FOLLOWERS
- DRIVER: High outgoing causality, low incoming. Monitor these closely.
- FOLLOWER: High incoming causality, low outgoing. Responds to others.
- NEUTRAL: Balanced. Part of feedback loops.

═══════════════════════════════════════════════════════════════════════════════
PART 7: PATTERN RECOGNITION
═══════════════════════════════════════════════════════════════════════════════

HEALTHY SYSTEM:
- High coherence (0.7+), stable over time
- Persistent memory in key variables
- Periodic signals where expected
- Low discontinuity
- Clear driver/follower structure

STRESS / PRE-FAILURE:
- Coherence declining or unstable
- Volatility increasing
- Increasing entropy
- Causal structure breaking down
- Discontinuities appearing

WARNING SEQUENCE:
1. Volatility increases in driver signal
2. Coherence starts dropping
3. Other signals become more entropic
4. Discontinuities appear
5. Failure / regime change

═══════════════════════════════════════════════════════════════════════════════
PART 8: DOMAIN TRANSLATION
═══════════════════════════════════════════════════════════════════════════════

INDUSTRIAL / MANUFACTURING:
- Coherence = "process stability"
- Driver = "root cause" or "leading indicator"
- Volatility clustering = "intermittent faults"
- Discontinuity = "trip" or "upset"

MEDICAL / PHYSIOLOGICAL:
- Coherence = "physiological coupling"
- High entropy = "HRV" (often good)
- Driver = "pacemaker" or "dominant oscillator"
- Volatility = "dysregulation"

CHEMISTRY / SPECTROSCOPY:
- Coherence = "spectral coherence"
- Discontinuity = "phase change"
- Driver wavelength = "indicator band"
- Regime change = "reaction phase"

═══════════════════════════════════════════════════════════════════════════════
PART 9: COMMUNICATION STYLE
═══════════════════════════════════════════════════════════════════════════════

BE SPECIFIC
- Use actual signal names: "HYD_PS1 drives HYD_PS2"
- Include numbers: "coherence dropped from 0.87 to 0.34 at t=500"

BE DIRECT
- Lead with the insight: "Your system has a clear driver: PS1"
- Don't hedge unnecessarily

BE HIERARCHICAL
1. Most critical finding (usually coherence/stability)
2. Causal structure (what drives what)
3. Group behavior
4. Individual signal characteristics

CONNECT TO ACTION
- "PS1 is the driver" → "Monitor PS1 for early warning"
- "Coherence dropped at t=500" → "Investigate what happened at t=500"

═══════════════════════════════════════════════════════════════════════════════
PART 10: YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

When given ORTHON results:

1. UNDERSTAND: Parse the results and identify key patterns
2. PRIORITIZE: What's most important? (Usually: stability + drivers)
3. EXPLAIN: Translate metrics into practical insight
4. CONNECT: Relate findings to each other
5. RECOMMEND: What should the user do? What should they monitor?

Always remember: The user has data but needs understanding. Your job is to bridge from numbers to insight.
'''


# =============================================================================
# ANALYSIS PROMPT TEMPLATE
# =============================================================================

ANALYSIS_PROMPT = '''
Analyze these ORTHON signal processing results:

DATASET: {name}
- {n_signals} signals
- {n_samples} samples
{domain_hint}

SIGNAL TYPOLOGY:
{typology_summary}

GROUPS ({n_clusters} clusters, silhouette={silhouette:.2f}):
{groups_summary}

DYNAMICS:
- Mean coherence: {mean_coherence:.2f}
- Range: {coherence_min:.2f} - {coherence_max:.2f}
{transitions_summary}

CAUSAL STRUCTURE:
- Drivers: {drivers}
- Followers: {followers}
- Causal density: {causal_density:.2f}
{causal_links_summary}

Provide a clear, specific analysis covering:
1. Overall system state (healthy/stressed/transitioning?)
2. Signal groupings and what distinguishes them
3. Key dynamics (stability, transitions)
4. Causal structure (what drives what)
5. What to monitor and why

Write 2-3 paragraphs. Be specific — use signal names and actual values.
'''


# =============================================================================
# CHAT SYSTEM TEMPLATE
# =============================================================================

CHAT_SYSTEM_TEMPLATE = '''
You are helping a user understand their ORTHON signal analysis.

You have access to the complete analysis results:

METADATA:
{metadata}

TYPOLOGY (per-signal characteristics):
{typology}

GROUPS:
{groups}

DYNAMICS:
{dynamics}

CAUSALITY:
{mechanics}

Answer questions about THIS SPECIFIC DATA. Be precise:
- Use actual signal names
- Reference actual values
- Cite specific time points

If the user asks something you cannot determine from the data, say so.
If they need more analysis, suggest what additional computation might help.

Remember the ORTHON framework:
- Typology = signal character (memory, volatility, etc.)
- Groups = behavioral clustering
- Dynamics = system evolution (especially coherence)
- Mechanics = causal relationships (Granger, transfer entropy)

Lead with practical insight. Support with numbers.
'''


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_client():
    """Get Anthropic client if available."""
    if not ANTHROPIC_AVAILABLE:
        return None
    try:
        return anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    except Exception:
        return None


def generate_analysis(results: dict, domain_hint: str = None) -> str:
    """
    Generate narrative analysis of computed results.

    Args:
        results: Dict with typology, groups, dynamics, mechanics
        domain_hint: Optional domain context ("hydraulic", "cardio", etc.)

    Returns:
        2-3 paragraph analysis
    """
    client = get_client()

    if client is None:
        return generate_fallback_analysis(results, domain_hint)

    prompt = format_results_for_prompt(results, domain_hint)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=ORTHON_ANALYST_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return generate_fallback_analysis(results, domain_hint)


def chat_about_data(
    question: str,
    results: dict,
    history: list = None
) -> str:
    """
    Answer follow-up questions about the analyzed data.

    Args:
        question: User's question
        results: Computed analysis results
        history: Previous conversation turns

    Returns:
        Claude's response
    """
    client = get_client()

    if client is None:
        return generate_fallback_chat(question, results)

    system = get_chat_system(results)

    messages = history or []
    messages.append({"role": "user", "content": question})

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception:
        return generate_fallback_chat(question, results)


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def format_results_for_prompt(results: dict, domain_hint: str = None) -> str:
    """Format results dict into analysis prompt."""

    metadata = results.get('metadata', {})
    typology = results.get('typology', [])
    groups = results.get('groups', {})
    dynamics = results.get('dynamics', {})
    mechanics = results.get('mechanics', {})

    # Typology summary
    typology_lines = []
    for sig in typology[:10]:  # Limit for prompt size
        traits = []
        if sig.get('memory', 0.5) > 0.6:
            traits.append('persistent')
        if sig.get('memory', 0.5) < 0.4:
            traits.append('forgetful')
        if sig.get('volatility', 0.5) > 0.6:
            traits.append('volatile')
        if sig.get('frequency', 0.5) > 0.6:
            traits.append('periodic')
        if sig.get('information', 0.5) > 0.6:
            traits.append('entropic')
        if sig.get('discontinuity', 0.5) > 0.6:
            traits.append('step-like')
        trait_str = ', '.join(traits) if traits else 'neutral'
        typology_lines.append(f"- {sig.get('signal_id', 'unknown')}: {trait_str}")

    if len(typology) > 10:
        typology_lines.append(f"- ... and {len(typology) - 10} more signals")

    # Groups summary
    groups_lines = []
    for cluster in groups.get('clusters', []):
        members = cluster.get('members', [])
        member_str = ', '.join(members[:5])
        if len(members) > 5:
            member_str += f" (+{len(members)-5})"
        groups_lines.append(
            f"- Group {cluster.get('id', '?')}: [{member_str}] — {cluster.get('dominant_trait', 'mixed')}"
        )

    # Transitions summary
    transitions = dynamics.get('transitions', [])
    if transitions:
        trans_lines = [
            f"- t={t.get('time', '?')}: coherence {t.get('from_coherence', 0):.2f} → {t.get('to_coherence', 0):.2f}"
            for t in transitions[:3]
        ]
        transitions_summary = "Transitions:\n" + '\n'.join(trans_lines)
    else:
        transitions_summary = "No significant transitions detected"

    # Causal links summary
    links = mechanics.get('top_links', [])[:5]
    if links:
        link_lines = [
            f"- {l.get('source', '?')} → {l.get('target', '?')}: F={l.get('granger_f', 0):.1f}, TE={l.get('transfer_entropy', 0):.2f} bits"
            for l in links
        ]
        causal_links_summary = '\n'.join(link_lines)
    else:
        causal_links_summary = "No strong causal links detected"

    return ANALYSIS_PROMPT.format(
        name=metadata.get('name', 'Unknown'),
        n_signals=metadata.get('n_signals', len(typology)),
        n_samples=metadata.get('n_samples', 0),
        domain_hint=f"- Domain: {domain_hint}" if domain_hint else "",
        typology_summary='\n'.join(typology_lines) if typology_lines else "No typology data",
        n_clusters=groups.get('n_clusters', 0),
        silhouette=groups.get('silhouette', 0),
        groups_summary='\n'.join(groups_lines) if groups_lines else "No groups detected",
        mean_coherence=dynamics.get('mean_coherence', 0.5),
        coherence_min=dynamics.get('coherence_min', 0),
        coherence_max=dynamics.get('coherence_max', 1),
        transitions_summary=transitions_summary,
        drivers=', '.join(mechanics.get('drivers', [])) or 'None identified',
        followers=', '.join(mechanics.get('followers', [])) or 'None identified',
        causal_density=mechanics.get('causal_density', 0),
        causal_links_summary=causal_links_summary,
    )


def get_chat_system(results: dict) -> str:
    """Build chat system prompt with full results context."""
    return CHAT_SYSTEM_TEMPLATE.format(
        metadata=json.dumps(results.get('metadata', {}), indent=2),
        typology=json.dumps(results.get('typology', [])[:10], indent=2),
        groups=json.dumps(results.get('groups', {}), indent=2),
        dynamics=json.dumps(results.get('dynamics', {}), indent=2),
        mechanics=json.dumps(results.get('mechanics', {}), indent=2),
    )


# =============================================================================
# FALLBACK FUNCTIONS (when Claude API unavailable)
# =============================================================================

def generate_fallback_analysis(results: dict, domain_hint: str = None) -> str:
    """
    Generate analysis without Claude API (fallback).
    Creates a structured summary from the computed metrics.
    """
    metadata = results.get('metadata', {})
    typology = results.get('typology', [])
    groups = results.get('groups', {})
    dynamics = results.get('dynamics', {})
    mechanics = results.get('mechanics', {})

    name = metadata.get('name', 'your dataset')
    n_signals = metadata.get('n_signals', len(typology))
    n_samples = metadata.get('n_samples', 0)

    paragraphs = []

    # Paragraph 1: Typology and groups
    n_clusters = groups.get('n_clusters', 0)
    if n_clusters > 1:
        para1 = f"Your {name} contains {n_signals} signals that naturally organize into {n_clusters} distinct behavioral groups. "
    else:
        para1 = f"Your {name} contains {n_signals} signals. "

    # Describe dominant traits
    trait_counts = {}
    for sig in typology:
        trait = sig.get('dominant_trait', 'unknown')
        trait_counts[trait] = trait_counts.get(trait, 0) + 1

    if trait_counts:
        top_traits = sorted(trait_counts.items(), key=lambda x: -x[1])[:3]
        trait_desc = ", ".join([f"{count} {trait}" for trait, count in top_traits])
        para1 += f"The signals show diverse characteristics: {trait_desc}."

    paragraphs.append(para1)

    # Paragraph 2: Dynamics
    mean_coh = dynamics.get('mean_coherence', 0.5)
    transitions = dynamics.get('transitions', [])

    if mean_coh > 0.7:
        para2 = "The system shows strong coherence — signals move together, indicating healthy coupling. "
    elif mean_coh < 0.4:
        para2 = "The system shows weak coherence — signals behave independently, which may indicate decoupling or stress. "
    else:
        para2 = "The system shows moderate coherence with some coordination between signals. "

    if transitions:
        t = transitions[0]
        para2 += f"A significant transition occurred around sample {t.get('time', 'unknown')}, "
        para2 += f"where coherence shifted from {t.get('from_coherence', 0):.2f} to {t.get('to_coherence', 0):.2f}. This warrants investigation."
    else:
        para2 += "No major regime transitions were detected — the system maintained stable behavior."

    paragraphs.append(para2)

    # Paragraph 3: Causality
    drivers = mechanics.get('drivers', [])
    followers = mechanics.get('followers', [])

    if drivers:
        para3 = f"Causal analysis identifies **{drivers[0]}** as a primary driver in the system"
        if len(drivers) > 1:
            para3 += f", along with {', '.join(drivers[1:])}. "
        else:
            para3 += ". "

        n_caused = count_caused(mechanics, drivers[0])
        if n_caused > 0:
            para3 += f"It influences {n_caused} other signals — monitor this signal closely for early warning signs. "

        if followers:
            para3 += f"The main followers ({', '.join(followers[:2])}) respond to changes elsewhere in the system."
    else:
        para3 = "No strong causal drivers were identified — the signals may be responding to external factors or operating independently."

    paragraphs.append(para3)

    return "\n\n".join(paragraphs)


def generate_fallback_chat(question: str, results: dict) -> str:
    """Fallback chat response without Claude API."""

    question_lower = question.lower()

    # Keyword-based responses
    if 'driver' in question_lower or 'cause' in question_lower or 'granger' in question_lower:
        drivers = results.get('mechanics', {}).get('drivers', [])
        if drivers:
            return f"Based on Granger causality analysis, **{drivers[0]}** appears to be the primary driver in this system. It shows predictive power over other signals, meaning its past values help forecast their future behavior. Monitor this signal closely — changes here often propagate to the rest of the system."
        return "No strong causal drivers were identified in this dataset. The signals may be responding to external factors rather than each other, or the causal relationships may be too weak to detect reliably."

    if 'group' in question_lower or 'cluster' in question_lower:
        groups = results.get('groups', {})
        n = groups.get('n_clusters', 0)
        silhouette = groups.get('silhouette', 0)
        if n > 1:
            return f"The signals cluster into **{n} distinct groups** based on their behavioral characteristics (silhouette score: {silhouette:.2f}). Signals in the same group share similar typology profiles — they tend to have similar memory, volatility, and periodicity patterns. This grouping often reflects physical subsystems or functional relationships in the underlying process."
        return "The signals don't form distinct clusters — they share similar behavioral characteristics across the board."

    if 'transition' in question_lower or 'change' in question_lower or 'coherence' in question_lower:
        transitions = results.get('dynamics', {}).get('transitions', [])
        mean_coh = results.get('dynamics', {}).get('mean_coherence', 0.5)
        if transitions:
            t = transitions[0]
            return f"A significant transition was detected around sample **{t.get('time', 'unknown')}**. The system coherence dropped from {t.get('from_coherence', 0):.2f} to {t.get('to_coherence', 0):.2f}, indicating the signals became less synchronized. This type of coherence breakdown often precedes or accompanies regime changes, fault conditions, or external disturbances. Investigate what was happening at this time point."
        return f"No major transitions were detected. The system maintained relatively stable behavior with mean coherence of {mean_coh:.2f} throughout the observation period."

    if 'memory' in question_lower or 'hurst' in question_lower:
        return "**Memory** (measured by the Hurst exponent) indicates how much a signal 'remembers' its past. A high memory score (>0.7) means the signal is **persistent** — trends tend to continue, and shocks have lasting effects. A low score (<0.3) means the signal is **forgetful** or mean-reverting — it quickly returns to baseline after disturbances. This is crucial for understanding how long-lasting the effects of interventions or faults will be."

    if 'volatility' in question_lower or 'garch' in question_lower:
        return "**Volatility clustering** (measured by GARCH) indicates whether periods of high variability tend to cluster together. High volatility scores suggest the signal shows bursts of activity followed by calm periods — common in systems with feedback loops or shock propagation. This is important for risk assessment: when volatility is clustered, one shock often predicts more shocks coming."

    if 'entropy' in question_lower or 'information' in question_lower:
        return "**Information/Entropy** measures how predictable or disordered a signal is. Low entropy signals are regular and predictable — easy to forecast. High entropy signals are complex or noisy — harder to predict. High entropy isn't necessarily bad; it can indicate rich, complex dynamics. But sudden increases in entropy often indicate a system becoming less stable or more chaotic."

    if 'monitor' in question_lower or 'watch' in question_lower or 'warning' in question_lower:
        drivers = results.get('mechanics', {}).get('drivers', [])
        if drivers:
            return f"For early warning, focus on the driver signals: **{', '.join(drivers)}**. These lead changes in the system, so monitoring them gives you advance notice. Also watch for: (1) coherence drops — signals decoupling is often a pre-failure indicator, (2) volatility increases in driver signals, and (3) discontinuities appearing where there were none before."
        return "Without clear causal drivers, monitor the signals with the highest volatility and those that show sudden changes in their typology (becoming more entropic, more volatile, or less periodic). Coherence is your best system-level indicator — watch for drops."

    # Default response
    return """I can help you understand the analysis results. Try asking about:
- **Drivers**: "Which signals drive the system?" / "What's the primary cause?"
- **Groups**: "How do the signals cluster?" / "What distinguishes the groups?"
- **Transitions**: "When did the system change?" / "What happened to coherence?"
- **Monitoring**: "What should I watch for early warning?"
- **Typology**: "What does memory/volatility/entropy mean?"

Or ask about specific signals by name."""


# =============================================================================
# INSIGHT CARD GENERATION
# =============================================================================

def generate_insight_cards(results: dict) -> List[Dict[str, Any]]:
    """
    Generate structured insight cards for UI display.

    Returns:
        List of insight cards with type, headline, detail, chart_type
    """
    cards = []

    # Groups card
    n_groups = results.get('groups', {}).get('n_clusters', 0)
    if n_groups > 1:
        silhouette = results.get('groups', {}).get('silhouette', 0)
        cards.append({
            'type': 'groups',
            'icon': '📊',
            'headline': f'{n_groups} Signal Groups',
            'detail': f"Well-separated clusters (silhouette: {silhouette:.2f})" if silhouette > 0.5 else describe_groups_brief(results.get('groups', {})),
            'chart_type': 'scatter',
            'link_to': 'Groups',
        })

    # Transition card
    transitions = results.get('dynamics', {}).get('transitions', [])
    if transitions:
        t = transitions[0]
        cards.append({
            'type': 'transition',
            'icon': '⚡',
            'headline': f"Transition at t={t.get('time', '?')}",
            'detail': f"Coherence: {t.get('from_coherence', 0):.2f} → {t.get('to_coherence', 0):.2f}",
            'chart_type': 'line',
            'link_to': 'Dynamics',
        })

    # Driver card
    drivers = results.get('mechanics', {}).get('drivers', [])
    if drivers:
        n_caused = count_caused(results.get('mechanics', {}), drivers[0])
        cards.append({
            'type': 'causality',
            'icon': '🎯',
            'headline': f"{drivers[0]} Drives System",
            'detail': f"Granger-causes {n_caused} signals" if n_caused else "Primary causal influence",
            'chart_type': 'network',
            'link_to': 'Mechanics',
        })

    # Anomaly card
    anomalies = find_anomalous_signals(results.get('typology', []))
    if anomalies:
        cards.append({
            'type': 'anomaly',
            'icon': '🚨',
            'headline': f"{anomalies[0]['signal']} is Unusual",
            'detail': anomalies[0]['reason'],
            'chart_type': 'radar',
            'link_to': 'Typology',
        })

    # Coherence card (if no transition but notable coherence)
    if not transitions and len(cards) < 3:
        mean_coh = results.get('dynamics', {}).get('mean_coherence', 0.5)
        if mean_coh > 0.7:
            cards.append({
                'type': 'coherence',
                'icon': '🔗',
                'headline': 'Strong System Coupling',
                'detail': f"Mean coherence: {mean_coh:.2f}",
                'chart_type': 'line',
                'link_to': 'Dynamics',
            })
        elif mean_coh < 0.4:
            cards.append({
                'type': 'coherence',
                'icon': '⚠️',
                'headline': 'Weak System Coupling',
                'detail': f"Mean coherence: {mean_coh:.2f} — signals decoupled",
                'chart_type': 'line',
                'link_to': 'Dynamics',
            })

    return cards[:3]  # Max 3 cards on discovery page


def describe_groups_brief(groups: dict) -> str:
    """One-line group description for card."""
    clusters = groups.get('clusters', [])
    if not clusters:
        return "No distinct clusters"
    cluster_sizes = [len(c.get('members', [])) for c in clusters]
    return f"Clusters of {', '.join(map(str, cluster_sizes))} signals"


def count_caused(mechanics: dict, driver: str) -> int:
    """Count how many signals a driver causes."""
    return sum(1 for link in mechanics.get('top_links', [])
               if link.get('source') == driver)


def find_anomalous_signals(typology: list) -> List[Dict]:
    """Find signals with unusual characteristics."""
    anomalies = []
    for sig in typology:
        distinctiveness = sig.get('distinctiveness', 0)
        if distinctiveness > 0.4:
            anomalies.append({
                'signal': sig.get('signal_id', 'unknown'),
                'reason': f"Extreme {sig.get('dominant_trait', 'characteristics')} (distinctiveness: {distinctiveness:.2f})",
                'distinctiveness': distinctiveness,
            })

    return sorted(anomalies, key=lambda x: -x.get('distinctiveness', 0))


def get_chat_suggestions(results: dict) -> List[str]:
    """Generate relevant chat suggestions based on results."""
    suggestions = []

    # Check for transitions
    transitions = results.get('dynamics', {}).get('transitions', [])
    if transitions:
        t = transitions[0]
        suggestions.append(f"Why did coherence drop at t={t.get('time', '?')}?")

    # Check for drivers
    drivers = results.get('mechanics', {}).get('drivers', [])
    if drivers:
        suggestions.append(f"What makes {drivers[0]} the primary driver?")

    # Check for groups
    n_groups = results.get('groups', {}).get('n_clusters', 0)
    if n_groups > 1:
        suggestions.append("What distinguishes the signal groups?")

    # Default suggestions
    suggestions.extend([
        "Which signals should I monitor for early warning?",
        "Is this system healthy or showing stress?",
    ])

    return suggestions[:4]
