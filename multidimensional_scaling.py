import numpy as np
import pandas as pd

from scipy.stats import pearsonr

import janitor

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS, smacof
from sklearn.metrics import pairwise_distances


def calculate_rsq(original_distances, embedded_distances):
    # Flatten distance matrices
    if original_distances.ndim > 1:
        original_flat = original_distances[
            np.triu_indices(original_distances.shape[0], k=1)
        ]
        embedded_flat = embedded_distances[
            np.triu_indices(embedded_distances.shape[0], k=1)
        ]
    else:
        original_flat = original_distances
        embedded_flat = embedded_distances

    correlation = np.corrcoef(original_flat, embedded_flat)[0, 1]
    return correlation**2


def create_voting_matrix(
    votes: pd.DataFrame,
    values: str,
    columns: str,
    index="aktørid",
    is_parliament_data=True,
) -> pd.DataFrame:
    """
    Creates a two-dimensional matrix of roll calls and agents, with their votes as values.
    Per default assumes data origin is from parliament, and replaces voting values with ones
    matching Altinget's candidate tests.
    5 = Completely agree, 1 = Completely disagree, 3 = Neither / nor and 0 = No response / absent.
    If the data is from Altinget or elsewhere, make sure to change the variable names to match.
    """
    votes = votes.copy()

    if is_parliament_data:
        votes[values] = votes[values].replace({1: 5, 2: 1, 3: 0, 4: 3})
        votes = janitor.complete(votes, columns, index, fill_value=3, explicit=False)

    voting_matrix = pd.pivot(votes, index=index, columns=columns, values=values)

    return voting_matrix


def normalise_coordinates(coordinate_dataframe: pd.DataFrame):

    normalised_df = coordinate_dataframe.copy()

    for col in normalised_df.columns:
        normalised_df[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
            normalised_df[[col]]
        )

    return normalised_df


def calculate_mds_dataframe(
    agents_and_roll_calls_matrix: pd.DataFrame, metric="manhattan", dimensions=1
) -> tuple:

    if dimensions not in [1, 2]:
        raise ValueError("Dimensions must be either 1 or 2")

    original_distances = pairwise_distances(agents_and_roll_calls_matrix, metric=metric)

    mds = MDS(
        n_components=dimensions,
        dissimilarity="precomputed",
        random_state=1312,
        n_init=10,
    )

    coordinates = mds.fit_transform(original_distances)

    embedded_distances = pairwise_distances(coordinates)

    fit_metrics = {
        "rsq": calculate_rsq(original_distances, embedded_distances),
    }

    if dimensions == 1:
        mds_dataframe = pd.DataFrame(
            {"x": coordinates.flatten()}, index=agents_and_roll_calls_matrix.index
        )
    else:
        mds_dataframe = pd.DataFrame(
            coordinates, columns=["x", "y"], index=agents_and_roll_calls_matrix.index
        )

    return mds_dataframe, fit_metrics


def calculate_smacof_dataframe(
    agents_and_roll_calls_matrix: pd.DataFrame, metric="manhattan", dimensions=1
) -> tuple:

    if dimensions not in [1, 2]:
        raise ValueError("Dimensions must be either 1 or 2")

    original_distances = pairwise_distances(agents_and_roll_calls_matrix, metric=metric)

    coordinates, stress = smacof(
        dissimilarities=original_distances,
        n_components=dimensions,
        n_init=100,
        random_state=1312,
    )

    embedded_distances = pairwise_distances(coordinates)

    fit_metrics = {
        "rsq": calculate_rsq(original_distances, embedded_distances),
    }

    if dimensions == 1:
        column_names = ["x"]
    else:
        column_names = ["x", "y"]

    smacof_df = pd.DataFrame(
        coordinates, index=agents_and_roll_calls_matrix.index, columns=column_names
    )

    return smacof_df, fit_metrics


def get_coordinates_and_fit_metrics(
    votes: pd.DataFrame,
    election_candidates: pd.DataFrame,
    return_normalised_coordinates: bool = True,
    return_basic_mds=False,
):
    """
    1. Takes dataframes containing voting- and election candidate test data and converts them to matrices.
    2. Performs multidimensional scaling on both matrices, by default SMACOF only but basic MDS can be toggled on.
    3. Calculates r-squared and pearson correlations for each combination of mds-approach and dataframe input.
    4. Optionally normalises coordinates to be between -1 and 1.
    5. Returns coordinates or "opinion vectors" and a dataframe containing correlation data.
    """

    matrix_votes = create_voting_matrix(votes, "typeid", "afstemningid")
    matrix_election = create_voting_matrix(
        election_candidates, "Answer", "Question", is_parliament_data=False
    )

    vote_coords_mds, vote_fit_metrics_mds = calculate_mds_dataframe(matrix_votes)
    election_coords_mds, election_fit_metrics_mds = calculate_mds_dataframe(
        matrix_election
    )

    vote_coords_smacof, vote_fit_metrics_smacof = calculate_smacof_dataframe(
        matrix_votes
    )
    election_coords_smacof, election_fit_metric_smacof = calculate_smacof_dataframe(
        matrix_election
    )

    mds = pd.concat(
        [
            pd.Series(election_fit_metrics_mds).rename(lambda k: f"{k} election"),
            pd.Series(vote_fit_metrics_mds).rename(lambda k: f"{k} votes"),
        ]
    )

    smacof = pd.concat(
        [
            pd.Series(election_fit_metric_smacof).rename(lambda k: f"{k} election"),
            pd.Series(vote_fit_metrics_smacof).rename(lambda k: f"{k} votes"),
        ]
    )

    matrix_fit_metrics = pd.concat([mds, smacof], axis=1)
    matrix_fit_metrics.columns = ["mds", "smacof"]
    matrix_fit_metrics = matrix_fit_metrics.round(4)

    coords = [vote_coords_smacof, election_coords_smacof]

    # I know the metrics are better for SMACOF, so no need to save this
    if return_basic_mds:
        coords = [
            vote_coords_mds,
            election_coords_mds,
            vote_coords_smacof,
            election_coords_smacof,
        ]

    if return_normalised_coordinates:
        coords = [normalise_coordinates(coord) for coord in coords]

    return (*coords, matrix_fit_metrics)
