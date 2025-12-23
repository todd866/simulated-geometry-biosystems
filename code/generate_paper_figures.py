"""
Generate figures for "Simulated Geometry" paper.

Figure 1: The Information/Geometry Distinction
Figure 2: Match Error Phase Space (AI vs Biology)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2


def figure1_information_vs_geometry():
    """
    Conceptual figure showing the core distinction.
    Left: Information (copyable, substrate-independent)
    Right: Geometry (instantiated, substrate-dependent)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: Information
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('INFORMATION', fontsize=14, fontweight='bold', pad=20)

    # Draw "copyable" symbols - multiple identical copies
    for i, (x, y) in enumerate([(2, 7), (5, 7), (8, 7)]):
        box = FancyBboxPatch((x-0.8, y-0.8), 1.6, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, '101', ha='center', va='center', fontsize=12,
                fontfamily='monospace', fontweight='bold')

    # Arrows showing copying
    ax1.annotate('', xy=(4, 7), xytext=(3, 7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax1.annotate('', xy=(7, 7), xytext=(6, 7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Labels
    ax1.text(5, 4.5, 'Substrate-independent', ha='center', fontsize=11, style='italic')
    ax1.text(5, 3.5, 'Can be copied without loss', ha='center', fontsize=10)
    ax1.text(5, 2.5, 'Shannon entropy, bits, facts', ha='center', fontsize=10, color='gray')

    # Examples
    ax1.text(5, 1.2, 'Examples: genome sequence, weights, database',
             ha='center', fontsize=9, color='darkblue')

    # Right panel: Geometry
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('GEOMETRY', fontsize=14, fontweight='bold', pad=20)

    # Draw a dynamic system - interconnected oscillating nodes
    center = (5, 6.5)
    radius = 2
    n_nodes = 6
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)

    # Draw connections first (behind nodes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            x1, y1 = center[0] + radius*np.cos(angles[i]), center[1] + radius*np.sin(angles[i])
            x2, y2 = center[0] + radius*np.cos(angles[j]), center[1] + radius*np.sin(angles[j])
            ax2.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1)

    # Draw nodes with oscillation indicators
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, n_nodes))
    for i, angle in enumerate(angles):
        x, y = center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle)
        circle = Circle((x, y), 0.35, facecolor=colors[i], edgecolor='darkred', linewidth=2)
        ax2.add_patch(circle)
        # Add oscillation wave
        wave_x = np.linspace(x-0.3, x+0.3, 20)
        wave_y = y + 0.5 + 0.15*np.sin(8*np.pi*(wave_x-x)/0.6 + i)
        ax2.plot(wave_x, wave_y, 'darkred', linewidth=1, alpha=0.7)

    # Central coupling indicator
    ax2.text(center[0], center[1], '↔', ha='center', va='center', fontsize=16, color='darkred')

    # Labels
    ax2.text(5, 3.2, 'Substrate-dependent', ha='center', fontsize=11, style='italic')
    ax2.text(5, 2.2, 'Must be instantiated & maintained', ha='center', fontsize=10)
    ax2.text(5, 1.2, 'Constraint manifold, cross-scale coupling', ha='center', fontsize=10, color='gray')

    # Examples
    ax2.text(5, 0.2, 'Examples: living cell, beating heart, neural dynamics',
             ha='center', fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig('../figures/figure1_distinction.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/figure1_distinction.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved figure1_distinction.pdf/png")


def figure2_match_error_phase_space():
    """
    Phase space showing D_geom vs R_spec with different regimes.
    Biology clusters near diagonal (matched), AI in corner (mismatched).
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    np.random.seed(42)

    # Generate synthetic data points for different regimes

    # Healthy biology: matched (along diagonal with scatter)
    n_healthy = 25
    healthy_base = np.random.uniform(4, 12, n_healthy)
    healthy_geom = healthy_base + np.random.normal(0, 0.8, n_healthy)
    healthy_spec = healthy_base + np.random.normal(0, 0.8, n_healthy)

    # Pathological: mismatched (off diagonal)
    n_pathology = 15
    pathology_geom = np.random.uniform(3, 8, n_pathology)
    pathology_spec = pathology_geom * np.random.uniform(0.3, 0.6, n_pathology)

    # Death/collapse: both low
    n_death = 10
    death_geom = np.random.uniform(0.5, 2, n_death)
    death_spec = np.random.uniform(0.3, 1.5, n_death)

    # AI simulation: high geom, low spec
    n_ai = 20
    ai_geom = np.random.uniform(12, 18, n_ai)
    ai_spec = np.random.uniform(1, 4, n_ai)

    # Plot
    ax.scatter(healthy_geom, healthy_spec, c='forestgreen', s=80, alpha=0.7,
               edgecolors='darkgreen', linewidths=1, label='Healthy Biology', zorder=3)
    ax.scatter(pathology_geom, pathology_spec, c='orange', s=80, alpha=0.7,
               edgecolors='darkorange', linewidths=1, label='Pathology', zorder=3)
    ax.scatter(death_geom, death_spec, c='gray', s=80, alpha=0.7,
               edgecolors='black', linewidths=1, label='Death/Collapse', zorder=3)
    ax.scatter(ai_geom, ai_spec, c='purple', s=80, alpha=0.7, marker='s',
               edgecolors='indigo', linewidths=1, label='AI Simulation', zorder=3)

    # Diagonal line (perfect match)
    ax.plot([0, 20], [0, 20], 'k--', alpha=0.3, linewidth=2, label='Perfect Match (ε=0)')

    # Shaded regions
    # Healthy region
    from matplotlib.patches import Polygon
    healthy_region = Polygon([(3, 3), (14, 14), (14, 10), (10, 3)],
                             alpha=0.1, facecolor='green', edgecolor='none')
    ax.add_patch(healthy_region)

    # AI region
    ai_region = Polygon([(11, 0), (20, 0), (20, 5), (11, 5)],
                        alpha=0.1, facecolor='purple', edgecolor='none')
    ax.add_patch(ai_region)

    # Collapse region
    collapse_region = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)],
                              alpha=0.1, facecolor='gray', edgecolor='none')
    ax.add_patch(collapse_region)

    # Labels for regions
    ax.text(8, 11, 'MATCHED\n(low ε)', ha='center', va='center',
            fontsize=10, color='darkgreen', fontweight='bold')
    ax.text(15.5, 2.5, 'DECOUPLED\n(high ε)', ha='center', va='center',
            fontsize=10, color='indigo', fontweight='bold')
    ax.text(1.5, 1.5, 'COLLAPSED', ha='center', va='center',
            fontsize=9, color='dimgray', fontweight='bold')

    # Axis labels
    ax.set_xlabel('Geometric Complexity ($D_{geom}$)', fontsize=12)
    ax.set_ylabel('Spectral Richness ($R_{spec}$)', fontsize=12)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)

    # Legend
    ax.legend(loc='upper left', framealpha=0.9)

    # Title
    ax.set_title('Match Error Phase Space', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate('AI: High information,\nno intrinsic dynamics',
                xy=(15, 3), xytext=(14, 8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.annotate('Life: Geometry and\ndynamics coupled',
                xy=(9, 9), xytext=(4, 13),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../figures/figure2_phase_space.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/figure2_phase_space.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved figure2_phase_space.pdf/png")


def figure3_dormancy_spectrum():
    """
    Shows the dormancy-complexity relationship.
    X-axis: Geometric complexity (dimensionality)
    Y-axis: Dormancy viability
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Trend line FIRST (so it's behind everything)
    x_trend = np.linspace(0.5, 15.5, 100)
    y_trend = 0.98 / (1 + 0.08 * x_trend**1.8)
    ax.plot(x_trend, y_trend, 'k--', alpha=0.3, linewidth=2, zorder=1)

    # Data points with labels ABOVE the curve (positive y offset from point)
    # (name, x, y, color)
    organisms = [
        ('Germ cells', 1.2, 0.90, 'purple'),
        ('Spores', 3.0, 0.72, 'olive'),
        ('Seeds', 5.0, 0.52, 'saddlebrown'),
        ('Tardigrades', 7.0, 0.34, 'teal'),
        ('C. elegans', 9.0, 0.20, 'coral'),
        ('Fish embryos', 11.0, 0.10, 'steelblue'),
        ('Mammalian cells', 13.0, 0.05, 'crimson'),
        ('Whole mammals', 15.0, 0.02, 'darkred'),
    ]

    # Plot points and labels
    for name, x, y, color in organisms:
        ax.scatter(x, y, s=120, c=color, edgecolors='black', linewidths=1.5, zorder=3)
        # All labels above the point
        ax.text(x, y + 0.10, name, ha='center', va='bottom', fontsize=8,
                fontweight='bold', color=color)

    ax.set_xlabel('Geometric Complexity', fontsize=11)
    ax.set_ylabel('Dormancy Viability', fontsize=11)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.05, 1.10)
    ax.set_title('Dormancy Difficulty Scales with Dimensionality', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    plt.savefig('../figures/figure3_dormancy.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/figure3_dormancy.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved figure3_dormancy.pdf/png")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Generating paper figures...")
    print("=" * 50)

    figure1_information_vs_geometry()
    figure2_match_error_phase_space()
    figure3_dormancy_spectrum()

    print("=" * 50)
    print("Done!")
