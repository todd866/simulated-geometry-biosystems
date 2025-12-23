"""
Generate all figures for the Dimension Matching paper.

Figure 1: GMC measures at different gamma values (subcritical to near-critical)
Figure 2: Dimension matching plot (D_C vs D_F across gamma) with theory curve
Figure 3: Phase diagram / regime schematic
Figure 4: Game-theoretic interpretation schematic
Figure 5: Information vs Dimension conceptual diagram

Note: Critical γ = √2 ≈ 1.414 (not 2!)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from gmc_simulation import (
    generate_log_correlated_field,
    construct_gmc_measure,
    estimate_correlation_dimension,
    estimate_fourier_dimension,
    simulate_gmc_ensemble,
    theory_dimension,
    GAMMA_CRITICAL
)

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def figure1_gmc_measures():
    """
    Figure 1: GMC measures at different gamma values.

    Shows how the measure becomes increasingly concentrated (spiky)
    as gamma approaches the critical value of √2 ≈ 1.414.
    """
    print("Generating Figure 1: GMC measures...")

    n_points = 4096
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Subcritical values only (γ < √2 ≈ 1.414)
    gammas = [0.3, 0.6, 0.9, 1.2]
    labels = [f'γ={g}, D*={theory_dimension(g):.2f}' for g in gammas]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Use same underlying field for comparison
    X = generate_log_correlated_field(n_points, n_modes=800, seed=42)

    for ax, gamma, label in zip(axes.flat, gammas, labels):
        mu = construct_gmc_measure(X, gamma, normalize=True)

        # Plot measure
        ax.fill_between(theta, 0, mu * n_points, alpha=0.4, color='steelblue')
        ax.plot(theta, mu * n_points, 'b-', linewidth=0.8)

        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('θ')
        ax.set_ylabel('μ_γ density')
        ax.set_xlim(0, 2*np.pi)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

        # Add statistics
        max_val = np.max(mu * n_points)
        participation = 1.0 / np.sum(mu**2) / n_points
        ax.text(0.95, 0.95, f'PR = {participation:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Gaussian Multiplicative Chaos: Subcritical Regime (γ < √2)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig1_gmc_measures.png')
    plt.savefig('../figures/fig1_gmc_measures.pdf')
    print("  Saved fig1_gmc_measures.png/pdf")
    plt.close()


def figure2_dimension_matching():
    """
    Figure 2: Dimension matching across gamma values.

    Shows D_C ≈ D_F in subcritical regime (γ < √2), with theory curve.
    """
    print("Generating Figure 2: Dimension matching...")

    # Simulate across gamma values (subcritical only)
    gamma_values = np.linspace(0.1, 1.35, 15)  # Stay below √2 ≈ 1.414
    results = simulate_gmc_ensemble(
        gamma_values,
        n_realizations=15,
        n_points=4096,
        n_modes=600
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: D_C and D_F vs gamma with theory curve
    ax1 = axes[0]

    # Theory curve
    gamma_fine = np.linspace(0.01, 1.4, 100)
    D_theory_fine = [theory_dimension(g) for g in gamma_fine]
    ax1.plot(gamma_fine, D_theory_fine, 'k-', linewidth=2, label='Theory $D^*(\\gamma)$')

    ax1.errorbar(results['gamma'], results['D_C'],
                 yerr=results['D_C_std'], fmt='o', color='steelblue',
                 label='$D_C$ (estimated)', capsize=3, markersize=6, alpha=0.8)
    ax1.errorbar(results['gamma'], results['D_F'],
                 yerr=results['D_F_std'], fmt='s', color='coral',
                 label='$D_F$ (estimated)', capsize=3, markersize=6, alpha=0.8)

    # Mark critical point
    ax1.axvline(x=GAMMA_CRITICAL, color='red', linestyle='--', alpha=0.5,
                label=f'Critical (γ=√2≈{GAMMA_CRITICAL:.2f})')
    ax1.axvline(x=1.0/np.sqrt(2), color='gray', linestyle=':', alpha=0.5,
                label=f'Transition (γ=1/√2≈{1/np.sqrt(2):.2f})')

    ax1.set_xlabel('GMC parameter γ')
    ax1.set_ylabel('Dimension')
    ax1.set_title('(A) Dimensions vs. γ', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # Panel B: D_C vs D_F scatter showing matching
    ax2 = axes[1]

    # Color by gamma
    gamma_norm = (results['gamma'] - 0.1) / 1.3
    colors = plt.cm.viridis(gamma_norm)
    for i, (dc, df, g) in enumerate(zip(results['D_C'], results['D_F'], results['gamma'])):
        if not np.isnan(dc) and not np.isnan(df):
            ax2.scatter(dc, df, c=[colors[i]], s=80, edgecolors='k', linewidths=0.5)

    # Add diagonal line (perfect matching)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='$D_C = D_F$')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0.1, 1.35))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, label='γ')

    ax2.set_xlabel('Correlation dimension $D_C$')
    ax2.set_ylabel('Fourier dimension $D_F$')
    ax2.set_title('(B) Dimension Matching', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig2_dimension_matching.png')
    plt.savefig('../figures/fig2_dimension_matching.pdf')
    print("  Saved fig2_dimension_matching.png/pdf")
    plt.close()

    return results


def figure3_phase_diagram():
    """
    Figure 3: Conceptual phase diagram.

    Shows regimes: incoherent, coherent (dimension matching), collapsed.
    """
    print("Generating Figure 3: Phase diagram...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background regions
    # Incoherent (low coupling, high noise)
    ax.fill([0, 0.4, 0.4, 0], [0.6, 0.6, 1, 1],
            color='lightgray', alpha=0.5, label='Incoherent')

    # Coherent (middle region - dimension matching)
    ax.fill([0.4, 0.85, 0.85, 0.4], [0, 0, 1, 1],
            color='lightblue', alpha=0.5, label='Coherent\n($D_C = D_F$)')

    # Collapsed (high gamma / strong noise)
    ax.fill([0.85, 1, 1, 0.85], [0, 0, 1, 1],
            color='lightsalmon', alpha=0.5, label='Collapsed')

    # Critical line
    ax.axvline(x=0.85, color='red', linewidth=2, linestyle='-', label='Phase transition')

    # Dimension matching zone annotation
    ax.annotate('Dimension\nMatching\nZone',
                xy=(0.6, 0.5), fontsize=14, ha='center', va='center',
                fontweight='bold', color='darkblue')

    # Add arrows showing transitions
    ax.annotate('', xy=(0.9, 0.5), xytext=(0.8, 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.92, 0.5, 'Collapse', fontsize=10, va='center', color='darkred')

    # Axes labels
    ax.set_xlabel('Noise strength / γ', fontsize=12)
    ax.set_ylabel('Scale coupling strength', fontsize=12)
    ax.set_title('Phase Diagram: Coherence and Collapse in Multiscale Systems',
                 fontsize=14, fontweight='bold')

    # Custom legend
    handles = [
        mpatches.Patch(color='lightgray', alpha=0.5, label='Incoherent'),
        mpatches.Patch(color='lightblue', alpha=0.5, label='Coherent ($D_C = D_F$)'),
        mpatches.Patch(color='lightsalmon', alpha=0.5, label='Collapsed'),
        Line2D([0], [0], color='red', linewidth=2, label='Critical transition')
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.4, 0.85, 1])
    ax.set_xticklabels(['0', 'Low', 'Critical', 'High'])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['Weak', 'Moderate', 'Strong'])

    plt.tight_layout()
    plt.savefig('../figures/fig3_phase_diagram.png')
    plt.savefig('../figures/fig3_phase_diagram.pdf')
    print("  Saved fig3_phase_diagram.png/pdf")
    plt.close()


def figure4_game_schematic():
    """
    Figure 4: Game-theoretic interpretation schematic.

    Shows scales as "players" with budget constraints.
    """
    print("Generating Figure 4: Game-theoretic schematic...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Cooperative equilibrium (subcritical)
    ax1 = axes[0]
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    # Setup
    n_scales = 5
    angles = np.linspace(0, 2*np.pi, n_scales, endpoint=False)
    radius = 0.8

    scale_labels = ['Scale 1\n(coarse)', 'Scale 2', 'Scale 3',
                    'Scale 4', 'Scale 5\n(fine)']
    colors = plt.cm.Blues(np.linspace(0.3, 0.8, n_scales))

    # Draw connections (pentagram) FIRST so they appear behind circles
    for i in range(n_scales):
        for j in range(i+1, n_scales):
            x1, y1 = radius * np.cos(angles[i]), radius * np.sin(angles[i])
            x2, y2 = radius * np.cos(angles[j]), radius * np.sin(angles[j])
            ax1.plot([x1, x2], [y1, y2], 'g-', linewidth=1.5, alpha=0.5, zorder=1)

    # Draw scales as circles ON TOP of connections
    for i, (angle, label, color) in enumerate(zip(angles, scale_labels, colors)):
        x, y = radius * np.cos(angle), radius * np.sin(angle)
        circle = Circle((x, y), 0.25, facecolor=color, edgecolor='black', linewidth=2, zorder=2)
        ax1.add_patch(circle)
        ax1.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # Center annotation
    ax1.text(0, 0, 'Martingale\nBalance', ha='center', va='center',
             fontsize=10, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    ax1.set_title('(A) Cooperative Regime: Dimension Matching', fontweight='bold')
    ax1.axis('off')
    ax1.set_aspect('equal')

    # Panel B: Collapse (one scale dominates)
    ax2 = axes[1]
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    # Draw scales - one dominates
    for i, (angle, label) in enumerate(zip(angles, scale_labels)):
        x, y = radius * np.cos(angle), radius * np.sin(angle)

        if i == 2:  # Dominant scale
            size = 0.45
            color = 'red'
            alpha = 1.0
        else:
            size = 0.15
            color = 'gray'
            alpha = 0.4

        circle = Circle((x, y), size, facecolor=color, edgecolor='black',
                        linewidth=1.5, alpha=alpha)
        ax2.add_patch(circle)

    # Arrows showing dominance
    dominant_x, dominant_y = radius * np.cos(angles[2]), radius * np.sin(angles[2])
    for i in range(n_scales):
        if i != 2:
            x, y = radius * np.cos(angles[i]), radius * np.sin(angles[i])
            dx, dy = (dominant_x - x) * 0.3, (dominant_y - y) * 0.3
            ax2.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # Center annotation
    ax2.text(0, -1.2, 'Scale 3 dominates\n→ Coherence lost', ha='center', va='center',
             fontsize=10, fontweight='bold', color='darkred',
             bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7))

    ax2.set_title('(B) Collapse: One Scale Dominates', fontweight='bold')
    ax2.axis('off')
    ax2.set_aspect('equal')

    plt.suptitle('Game-Theoretic View: Scales as Cooperative Agents',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/fig4_game_schematic.png')
    plt.savefig('../figures/fig4_game_schematic.pdf')
    print("  Saved fig4_game_schematic.png/pdf")
    plt.close()


def figure5_information_vs_dimension():
    """
    Figure 5: Information vs Dimension conceptual diagram.

    Shows that these are distinct axes.
    """
    print("Generating Figure 5: Information vs Dimension...")

    fig, ax = plt.subplots(figsize=(9, 7))

    # Axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Effective Dimension (geometric occupancy)', fontsize=12)
    ax.set_ylabel('Information Rate (entropy production)', fontsize=12)

    # Regions
    # Low-D, Low-Info: Fixed point
    ax.annotate('Fixed Point\n(trivial)', xy=(0.1, 0.1), fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))

    # Low-D, High-Info: Low-D chaos
    ax.annotate('Low-D Chaos\n(e.g., Lorenz)', xy=(0.15, 0.75), fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # High-D, Low-Info: Constrained high-D
    ax.annotate('High-D\nConstrained\n(thin manifold)', xy=(0.75, 0.15), fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    # Arrow: dimension matching trajectory (draw FIRST so boxes appear on top)
    ax.annotate('', xy=(0.68, 0.68), xytext=(0.2, 0.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                               connectionstyle='arc3,rad=0.15'),
                zorder=1)
    ax.text(0.52, 0.32, 'Dimension\nMatching\nRegime', fontsize=9, color='blue',
            rotation=0, ha='left', va='center', zorder=5)

    # High-D, High-Info: Complex / critical (draw AFTER arrow so it's on top)
    ax.annotate('Complex\nDynamics\n(coherent chaos)', xy=(0.7, 0.7), fontsize=10,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen', linewidth=1.5),
                zorder=10)

    # Pure noise corner
    ax.annotate('Pure Noise', xy=(0.9, 0.9), fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7),
                zorder=10)

    # Arrow: collapse
    ax.annotate('', xy=(0.9, 0.85), xytext=(0.75, 0.75),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.88, 0.78, 'Collapse', fontsize=9, color='red')

    ax.set_title('Information ≠ Dimension: Two Axes of Complexity',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../figures/fig5_info_vs_dimension.png')
    plt.savefig('../figures/fig5_info_vs_dimension.pdf')
    print("  Saved fig5_info_vs_dimension.png/pdf")
    plt.close()


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Generating figures for Dimension Matching paper")
    print("=" * 60)

    figure1_gmc_measures()
    results = figure2_dimension_matching()
    figure3_phase_diagram()
    figure4_game_schematic()
    figure5_information_vs_dimension()

    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)
