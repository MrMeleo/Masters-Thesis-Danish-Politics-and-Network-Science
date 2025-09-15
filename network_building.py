import warnings

# warnings.simplefilter(action = "ignore", category = DeprecationWarning)

import numpy as np
import pandas as pd

import networkx as nx

from scipy.stats import gaussian_kde
from scipy.stats import binom

from visualisations import plot_party_agreement_kdes

from itertools import combinations


def calculate_agreement_pair_counts(
    id_to_party: dict,
    votes: pd.DataFrame,
    roll_call_col: str = None,
    vote_type_col: str = None,
    agent_id_col: str = "aktørid",
) -> pd.DataFrame:
    """
    Calculates how many times each pairwise combination of politicians agree, including those with a count of 0.

    Returns:
    - DataFrame with columns ["src", "trg", "nij", same_party_boolean]
    """

    # Get all unique agents and create all possible pairs
    all_agents = sorted(votes[agent_id_col].unique())
    all_possible_pairs = pd.DataFrame(
        list(combinations(all_agents, 2)), columns=["src", "trg"]
    )

    # Calculate agreement counts by grouping votes and creating pairs within each group
    agreement_counts = (
        votes.groupby([roll_call_col, vote_type_col])[agent_id_col]
        .apply(lambda agents: pd.Series(list(combinations(sorted(agents), 2))))
        .reset_index(name="pair")
        .assign(src=lambda df: df["pair"].str[0], trg=lambda df: df["pair"].str[1])
        .groupby(["src", "trg"])
        .size()
        .reset_index(name="nij")
    )

    # Merge and add party information
    return all_possible_pairs.merge(
        agreement_counts, on=["src", "trg"], how="left"
    ).assign(
        nij=lambda df: df["nij"].fillna(0).astype(int),
        same_party=lambda df: df.apply(
            lambda row: id_to_party.get(row["src"]) == id_to_party.get(row["trg"]),
            axis=1,
        ),
    )


### Methods from backboning module by Michele Coscia: https://www.michelecoscia.com/?page_id=287 ###
def thresholding(table, threshold):
    """Reads a preprocessed edge table and returns only the edges supassing a significance threshold.

    Args:
    table (pandas.DataFrame): The edge table.
    threshold (float): The minimum significance to include the edge in the backbone.

    Returns:
    The network backbone.
    """
    table = table.copy()
    if "sdev_cij" in table:
        return table[(table["score"] - (threshold * table["sdev_cij"])) > 0][
            ["src", "trg", "nij", "score"]
        ]
    else:
        return table[table["score"] > threshold][["src", "trg", "nij", "score"]]


def test_densities(table, start, end, step):
    if start > end:
        raise ValueError("start must be lower than end")
    steps = []
    x = start
    while x <= end:
        steps.append(x)
        x += step
    onodes = len(set(table["src"]) | set(table["trg"]))
    oedges = table.shape[0]
    oavgdeg = (2.0 * oedges) / onodes
    for s in steps:
        edge_table = thresholding(table, s)
        nodes = len(set(edge_table["src"]) | set(edge_table["trg"]))
        edges = edge_table.shape[0]
        avgdeg = (2.0 * edges) / nodes
        yield (
            s,
            nodes,
            (100.0 * nodes) / onodes,
            edges,
            (100.0 * edges) / oedges,
            avgdeg,
            avgdeg / oavgdeg,
        )


def noise_corrected(
    table, undirected=False, return_self_loops=False, calculate_p_value=False
):
    table = table.copy()
    src_sum = table.groupby(by="src").sum()[["nij"]]
    table = table.merge(
        src_sum, left_on="src", right_index=True, suffixes=("", "_src_sum")
    )
    trg_sum = table.groupby(by="trg").sum()[["nij"]]
    table = table.merge(
        trg_sum, left_on="trg", right_index=True, suffixes=("", "_trg_sum")
    )
    table.rename(columns={"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace=True)
    table["n.."] = table["nij"].sum()
    table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (
        1 / table["n.."]
    )
    if calculate_p_value:
        table["score"] = binom.cdf(
            table["nij"], table["n.."], table["mean_prior_probability"]
        )
        return table[["src", "trg", "nij", "score"]]
    table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
    table["score"] = ((table["kappa"] * table["nij"]) - 1) / (
        (table["kappa"] * table["nij"]) + 1
    )
    table["var_prior_probability"] = (
        (1 / (table["n.."] ** 2))
        * (
            table["ni."]
            * table["n.j"]
            * (table["n.."] - table["ni."])
            * (table["n.."] - table["n.j"])
        )
        / ((table["n.."] ** 2) * ((table["n.."] - 1)))
    )
    table["alpha_prior"] = (
        ((table["mean_prior_probability"] ** 2) / table["var_prior_probability"])
        * (1 - table["mean_prior_probability"])
    ) - table["mean_prior_probability"]
    table["beta_prior"] = (
        table["mean_prior_probability"] / table["var_prior_probability"]
    ) * (1 - (table["mean_prior_probability"] ** 2)) - (
        1 - table["mean_prior_probability"]
    )
    table["alpha_post"] = table["alpha_prior"] + table["nij"]
    table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
    table["expected_pij"] = table["alpha_post"] / (
        table["alpha_post"] + table["beta_post"]
    )
    table["variance_nij"] = (
        table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
    )
    table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (
        table["n.."]
        * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2))
    )
    table["variance_cij"] = table["variance_nij"] * (
        (
            (2 * (table["kappa"] + (table["nij"] * table["d"])))
            / (((table["kappa"] * table["nij"]) + 1) ** 2)
        )
        ** 2
    )
    table["sdev_cij"] = table["variance_cij"] ** 0.5
    if not return_self_loops:
        table = table[table["src"] != table["trg"]]
    if undirected:
        table = table[table["src"] <= table["trg"]]
    return table[["src", "trg", "nij", "score", "sdev_cij"]]


# Adapted from reading a csv to reading a dataframe
def make_undirected(table: pd.DataFrame, drop_zeroes: bool = False):
    if drop_zeroes:
        table = table[table["nij"] > 0]
    table2 = table.copy()
    table2["new_src"] = table["trg"]
    table2["new_trg"] = table["src"]
    table2.drop(labels="src", axis=1, inplace=True)
    table2.drop(labels="trg", axis=1, inplace=True)
    table2 = table2.rename(columns={"new_src": "src", "new_trg": "trg"})
    table = pd.concat([table, table2], axis=0)
    table = table.drop_duplicates(subset=["src", "trg"])
    return table


### Largely unchanged from the implementation by Coscia: https://www.michelecoscia.com/?page_id=2105 ###
def find_intersection(
    kde1, kde2, scope=(0.1, 1), init_interval=0.01, convergence=0.0001
):
    def diff(x):
        return kde1(x)[0] - kde2(x)[0]

    # Find all intersections
    intersections = []
    x_left = scope[0]

    while x_left < scope[1]:
        x_right = x_left + init_interval
        if x_right > scope[1]:
            break

        if diff(x_left) * diff(x_right) < 0:
            # Found potential intersection, refine with binary search
            left, right = x_left, x_right
            while (right - left) > convergence:
                mid = (left + right) / 2
                if diff(left) * diff(mid) < 0:
                    right = mid
                else:
                    left = mid

            # Add the refined intersection point
            intersections.append((left + right) / 2)

        x_left = x_right

    # Return the highest intersection if any were found
    if intersections:
        return max(intersections)
    else:
        return None  # No intersection found


def get_kde_threshold(
    network_dataframe: pd.DataFrame, weight_col="nij", boolean_party_col="same_party"
):
    """
    Computes a threshold of agreement (co-votes) where it becomes more likely
    that two individuals are from the same party than from different ones.

    Parameters:
    - network_dataframe: DataFrame containing pairwise voting agreement data
    - weight_col: Column with co-vote counts or agreement weights
    - boolean_party_col: Column with boolean values indicating if the pair is from the same party

    Returns:
    - threshold: value of "weight_col" at which the two KDEs intersect
    - threshold_original_scale: same, but without normalisation
    """
    party_divided_dataframe = network_dataframe.copy()

    # Normalisation ensures compatibility between KDEs
    max_val = party_divided_dataframe[weight_col].max()
    party_divided_dataframe[weight_col] = party_divided_dataframe[weight_col] / max_val

    # Divides the dataframe by same-party and different-party
    party_divided_dataframe["is_same_party"] = np.where(
        party_divided_dataframe[boolean_party_col],
        party_divided_dataframe[weight_col],
        np.nan,
    )
    party_divided_dataframe["is_cross_party"] = np.where(
        ~party_divided_dataframe[boolean_party_col],
        party_divided_dataframe[weight_col],
        np.nan,
    )

    same_party_values = party_divided_dataframe["is_same_party"].dropna()
    different_party_values = party_divided_dataframe["is_cross_party"].dropna()

    same_party_kde = gaussian_kde(same_party_values)
    different_party_kde = gaussian_kde(different_party_values)

    threshold = find_intersection(same_party_kde, different_party_kde)
    plot_party_agreement_kdes(same_party_kde, different_party_kde, threshold)

    if threshold is None:
        print("Warning! No intersection found between the KDEs in the given scope.")

    threshold = round(threshold * max_val)
    print(f"KDE Threshold = {threshold}")
    return threshold


def get_nc_threshold(network, start, end, step):

    density_test_results = {}
    for test in test_densities(network, start, end, step):
        threshold = test[0]
        density_test_results[threshold] = test

    density_test_results = sorted(density_test_results.items())

    max_threshold = density_test_results[0][1][0]  # Initialize with first v[0]
    prev_amount_of_nodes = density_test_results[0][1][1]  # Initialize with first v[1]

    for test, results in density_test_results[1:]:
        current_amount_of_nodes = results[1]

        # If changing the threshold prunes a node, I return the threshold just before it
        if current_amount_of_nodes < prev_amount_of_nodes:
            print(f"NC Threshold = {max_threshold}")
            return max_threshold

        max_threshold = max(max_threshold, results[0])
        prev_amount_of_nodes = current_amount_of_nodes

    # If no prunes are found, returns the highest tested threshold.
    print(
        "No changes in amount of nodes found in the test results. A higher usable threshold may be available by increasing the end-parameter."
    )
    print(f"NC Threshold = {max_threshold}")
    return max_threshold


def backbone_kde_nc(
    processed_df: pd.DataFrame, kde_threshold: float, nc_threshold: float
):
    original_node_count = pd.concat(
        [processed_df["src"], processed_df["trg"]]
    ).nunique()
    original_edge_count = processed_df.shape[0]

    # This whole thing is an annoying but necessary failsafe since computational roundings may lead the NC to prune nodes by accident
    while True:
        temp_df = processed_df.copy()

        edges_removed_kde = temp_df[temp_df["nij"] < kde_threshold][
            ["src", "trg", "nij"]
        ]
        edges_removed_nc = temp_df[
            (temp_df["score"] - (nc_threshold * temp_df["sdev_cij"])) <= 0
        ][["src", "trg", "nij"]]

        intersection = pd.merge(
            left=edges_removed_kde, right=edges_removed_nc, how="inner"
        )

        temp_df = temp_df[
            ~temp_df[["src", "trg"]]
            .apply(tuple, axis=1)
            .isin(intersection[["src", "trg"]].apply(tuple, axis=1))
        ]

        new_node_count = pd.concat([temp_df["src"], temp_df["trg"]]).nunique()
        new_edge_count = temp_df.shape[0]

        graph = nx.from_pandas_edgelist(
            temp_df, source="src", target="trg", edge_attr="nij"
        )

        if nx.number_connected_components(graph) > 1:
            print(
                f"Warning: Network split into multiple components. Reducing NC threshold from {nc_threshold} to {nc_threshold - 0.1}"
            )
            nc_threshold -= 0.1
            continue

        if new_node_count != original_node_count:
            print(
                "Warning: One or more nodes has been removed! Double check threshold values."
            )

        print(
            f"Edges before: {original_edge_count}\n"
            f"Edges after: {new_edge_count}\n"
            f"Edges pruned: {original_edge_count - new_edge_count} "
            f"({round((1 - (new_edge_count / original_edge_count)) * 100, 1)}% of original amount)"
        )

        return temp_df


def determine_variable_values(dataframe_category: str):
    if "votes" in dataframe_category:
        roll_call = "afstemningid"
        vote_type = "typeid"
        start = -10
        end = 20
        step = 1
    elif "candidates" in dataframe_category:
        roll_call = "Question"
        vote_type = "Answer"
        start = -5
        end = 5
        step = 1
    else:
        raise NameError(
            "Please ensure the dataframe follows the naming convention 'votesFVxx' or 'candidatesFVxx'."
        )
    return roll_call, vote_type, start, end, step


def pipeline_dataframe_to_network(
    dataframe: pd.DataFrame, dataframe_category: str, id_to_party_map: dict
):

    roll_call, vote_type, start, end, step = determine_variable_values(
        dataframe_category
    )
    data = dataframe.copy()

    return (
        data.pipe(
            lambda df: calculate_agreement_pair_counts(
                id_to_party_map,
                df,
                roll_call,
                vote_type,
            )
        )
        .pipe(lambda df: make_undirected(df, drop_zeroes=True))
        .pipe(
            lambda df: pd.merge(
                noise_corrected(df, undirected=True),
                df[["src", "trg", "same_party"]],
                on=["src", "trg"],
                how="left",
            )
        )
        .pipe(
            lambda df: backbone_kde_nc(
                df, get_kde_threshold(df), get_nc_threshold(df, start, end, step)
            )
        )
    )
