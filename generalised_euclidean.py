import warnings

warnings.simplefilter(action="ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy import stats
from scipy.sparse import csgraph

from tqdm import tqdm


### Methods written by Michele Coscia: LINK ###
def calculate_Q(network, edge_weights=None):
    A = nx.adjacency_matrix(network, weight=edge_weights).todense().astype(float)
    return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed=False))


def ge(src, trg, network, edge_weights="nij", Q=None):
    """Calculate GE for network.

    Parameters:
    ----------
    srg: vector specifying node polarities
    trg: vector specifying node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError(
            """Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one."""
        )
    src = np.array([src[n] if n in src else 0.0 for n in network.nodes()])
    trg = np.array([trg[n] if n in trg else 0.0 for n in network.nodes()])
    diff = src - trg
    if Q is None:
        Q = calculate_Q(network, edge_weights=edge_weights)

    ge_dist = diff.T.dot(np.array(Q).dot(diff))

    if ge_dist < 0:
        ge_dist = 0
        print("Distance was below 0")
    return np.sqrt(ge_dist)


def split_opinion_vector(opinion_vector: pd.DataFrame):
    """
    Splits a dataframe column by positive and negative values.
    Outputs are used as src and trg when calculating the ge-distance.

    Parameters:
    ----------
    opinion_vector : a single vector in the form of a dataframe column

    Returns:
    --------
    positve, negative
        Tuple of postive values and absolute negative values.
    """
    positive = opinion_vector.clip(lower=0).iloc[:, 0]
    negative = -opinion_vector.clip(upper=0).iloc[:, 0]  # Gets the absolute value

    return positive, negative


def xtremify_party_coords(
    coords: pd.DataFrame, id_to_party: dict, coord_col="x", agent_id="aktørid"
):

    original_index = coords.index
    # Attach party info
    df = coords.copy()
    df = df.reset_index(names=agent_id).assign(
        party=lambda d: d[agent_id].map(id_to_party)
    )
    grouped = df.groupby("party")

    # Compute values - abs() finds distance from 0, but we keep original sign
    most_extreme = grouped.apply(
        lambda g: g.loc[g[coord_col].abs().idxmax(), coord_col]
    )
    least_extreme = grouped.apply(
        lambda g: g.loc[g[coord_col].abs().idxmin(), coord_col]
    )
    mean_value = grouped[coord_col].mean()

    # Global polarisation: find global min/max and polarise all values
    global_min = coords[coord_col].min()
    global_max = coords[coord_col].max()
    even_further_beyond = np.where(coords[coord_col] >= 0, global_max, global_min)

    # Map each value type back to the agents
    xtreme_series = df["party"].map(most_extreme)
    least_series = df["party"].map(least_extreme)
    mean_series = df["party"].map(mean_value)

    # Return 4 separate DataFrames with preserved index
    xtreme_df = pd.DataFrame({coord_col: xtreme_series.values}, index=original_index)
    least_df = pd.DataFrame({coord_col: least_series.values}, index=original_index)
    mean_df = pd.DataFrame({coord_col: mean_series.values}, index=original_index)
    even_further_beyond_df = pd.DataFrame(
        {coord_col: even_further_beyond}, index=original_index
    )

    return xtreme_df, least_df, mean_df, even_further_beyond_df


def contextualise_ge_result(opinion_vector, graph, id_to_party_map):
    xtreme, not_extreme, mean, polarised = xtremify_party_coords(
        opinion_vector, id_to_party_map
    )
    variants = {
        "original": opinion_vector,
        "least extreme": not_extreme,
        "party mean": mean,
        "most xtreme": xtreme,
        "polarised": polarised,
    }
    results = {}
    for variant, vector in variants.items():
        src, trg = split_opinion_vector(vector)
        result = ge(src, trg, graph)
        print(f"{variant} ge: {result}")
        results[variant] = result
    return results


# TODO rewrite in own words
def normalize_ge_results(results):
    """
    Normalize ge results so that least extreme = 0 and most extreme = 1

    Args:
        results: dict with keys 'original', 'most xtreme', 'least extreme', 'party mean'

    Returns:
        dict with normalized values
    """
    least_val = results["least extreme"]
    most_val = results["most xtreme"]

    if most_val == least_val:
        print("All ge values are identical - normalization not meaningful")
        return {k: 0.5 for k in results.keys()}  # Set all to middle value

    normalized = {}
    for variant, value in results.items():
        normalized[variant] = (value - least_val) / (most_val - least_val)

    print("\nNormalized results:")
    for variant, result in normalized.items():
        print(f"{variant} ge (normalized): {result}")

    return normalized
