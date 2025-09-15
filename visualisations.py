import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def plot_party_agreement_kdes(same_kde, diff_kde, threshold):
    """
    Plots KDEs for same-party and different-party co-vote distributions.

    Parameters:
    - same_kde: KDE of co-vote counts for same-party pairs
    - diff_vals: KDE of co-vote counts for different-party pairs
    - threshold: float marking the KDE intersection point
    """

    x = np.linspace(0, 1, 1000)
    same_y = same_kde(x)
    diff_y = diff_kde(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, same_y, label="Same Party", color="blue")
    plt.plot(x, diff_y, label="Different Party", color="red")

    if threshold is not None:
        plt.axvline(
            x=threshold,
            color="gray",
            linestyle="--",
            label=f"Threshold = {threshold:.2f}",
        )

    plt.title("KDE of Agreement Strength by Party")
    plt.xlabel("Normalised Co-Vote Count")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_normalised_outputs(results_df: pd.DataFrame, figsize=(8, 5)):
    """
    Visualise normalised outputs (0-1) for multiple result sets.
    Each row is a set of outputs (original, mean, min=0, max=1).
    """
    plot_data = []
    for label in results_df.index:
        for col in results_df.columns:
            plot_data.append(
                {"label": label, "type": col, "value": results_df.loc[label, col]}
            )
        # Add min and max for visual reference
        plot_data.append({"label": label, "type": "min", "value": 0})
        plot_data.append({"label": label, "type": "max", "value": 1})

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=figsize)

    # Plot min-max as horizontal lines
    for label in df["label"].unique():
        plt.plot([0, 1], [label, label], color="lightgray", linewidth=2, zorder=1)

    # Plot all types except min/max as points
    plot_types = [t for t in results_df.columns if t not in ("min", "max")]
    ax = sns.scatterplot(
        data=df[df["type"].isin(plot_types)],
        x="value",
        y="label",
        hue="type",
        s=200,
        zorder=2,
        legend="brief",
    )

    plt.xlabel("Normalised GE-score")
    plt.ylabel("Dataset")
    plt.title("Overview of normalised GE-scores")
    plt.xlim(-0.05, 1.05)

    # Move legend outside plot area (to the right)
    ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.tight_layout()
    plt.show()


def grouped_layout(
    group_dict, group_order=None, scale=6, fraction=5 / 8, rotation=270, seed=1312
):
    """
    Arrange nodes in clusters by group along an arc of a circle.

    Parameters
    ----------
    G : networkx.Graph
        The graph.
    group_dict : dict
        Mapping node -> group.
    group_order : list, optional
        Order of groups along the arc. If None, groups are sorted alphabetically.
    scale : float
        Radius of the arc.
    fraction : float
        Fraction of the circle to use (e.g., 2/3 for 240° arc).
    rotation : float
        Rotation angle in degrees to rotate the arc. Default is 90°.
    seed : int
        Random seed for jittering within clusters.

    Returns
    -------
    pos : dict
        Mapping node -> (x, y) position.
    """
    rng = np.random.default_rng(seed)

    # Organize nodes into groups
    groups = {}
    for node, group in group_dict.items():
        groups.setdefault(group, []).append(node)

    # Determine group order
    if group_order is None:
        group_order = sorted(groups.keys())

    n_groups = len(group_order)

    # Angular span of the arc
    arc_angle = 2 * np.pi * fraction

    # Center the arc around the x-axis initially (open upward)
    start_angle = -arc_angle / 2

    # Convert rotation to radians
    rot = np.deg2rad(rotation)

    pos = {}
    for i, group in enumerate(group_order):
        if group not in groups:
            continue
        # Evenly space groups along arc
        angle = start_angle + i * (arc_angle / (n_groups - 1 if n_groups > 1 else 1))
        # Apply rotation
        angle += rot
        center = np.array([scale * np.cos(angle), scale * np.sin(angle)])
        for node in groups[group]:
            pos[node] = center + 0.5 * rng.standard_normal(2)  # jitter inside cluster
    return pos


def plot_network_with_party_colors(
    G: nx.Graph,
    group_dict: dict,
    color_mapping: dict,
    party_order: list,
    figsize=(12, 8),
    node_size=75,
    same_color_alpha=0.6,
    diff_color_alpha=0.1,
    layout=None,
    label_offset=1.5,  # radial offset
    group_label_map=None,  # NEW: optional short labels
):
    """
    Plot a network with nodes colored by group and group labels placed outside clusters.

    Parameters
    ----------
    G : networkx.Graph
        The graph.
    group_dict : dict
        Mapping node -> group.
    color_mapping : dict
        Mapping group -> color.
    group_label_map : dict, optional
        Mapping full group name -> short label (for legend text).
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    if layout is None:
        pos = grouped_layout(group_dict, party_order)
    else:
        pos = layout

    # Node colors
    node_colors = [
        color_mapping.get(group_dict.get(node), "white") for node in G.nodes()
    ]

    # Create color lookup for edge processing
    node_color_map = dict(zip(G.nodes(), node_colors))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=1,
        ax=ax,
        edgecolors="black",
    )

    # Start with cross-party bucket
    edges_by_color = {"gray": []}
    for u, v in G.edges():
        if node_color_map[u] == node_color_map[v]:
            # Same party - group by color
            color = node_color_map[u]
            edges_by_color.setdefault(color, []).append((u, v))
        else:
            # Different parties - add to gray bucket
            edges_by_color["gray"].append((u, v))

    for color, edges in edges_by_color.items():
        if edges:  # Only draw if edges exist
            alpha = diff_color_alpha if color == "gray" else same_color_alpha
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color=color, alpha=alpha, ax=ax
            )

    if not group_label_map:
        return fig

    # Group nodes by party
    groups = {}
    for node, group in group_dict.items():
        groups.setdefault(group, []).append(node)

    for group, nodes in groups.items():
        # Calculate label position
        coords = np.array([pos[n] for n in nodes])
        center = coords.mean(axis=0)
        angle = np.arctan2(center[1], center[0])
        offset = np.array([np.cos(angle), np.sin(angle)]) * label_offset
        label_pos = center + offset

        # Use short label if available
        label_text = group_label_map.get(group, group) if group_label_map else group

        ax.text(
            label_pos[0],
            label_pos[1],
            str(label_text),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.3"
            ),
        )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
