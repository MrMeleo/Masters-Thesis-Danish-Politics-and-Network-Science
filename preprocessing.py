import os
import warnings
import pandas as pd

warnings.simplefilter(action="ignore", category=DeprecationWarning)


def get_procedures(
    year: str,
    government_dates: dict,
    file_path=(os.path.join(os.getcwd(), "Parliament CSV Files")),
) -> pd.DataFrame:
    start_date, end_date = government_dates[year]

    procedures_directory = os.path.join(file_path, "Sag.csv")
    procedural_steps_directory = os.path.join(file_path, "Sagstrin.csv")

    procedures = (
        pd.read_csv(procedures_directory, index_col="id", usecols=["id", "typeid"])
        .rename(
            columns={
                "id": "sagid",
                "typeid": "sagstypeid",
            }
        )
        .loc[lambda df: (df["sagstypeid"] == 3)]
    )

    return (
        pd.read_csv(
            procedural_steps_directory,
            low_memory=False,
            index_col="id",
            usecols=["id", "dato", "typeid", "sagid"],
            parse_dates=["dato"],
            date_format="%d-%m-%Y %H:%M:%S",
        )
        .rename(columns={"id": "sagstrinid", "typeid": "sagstrintypeid"})
        .pipe(
            lambda df: df.loc[
                df["dato"].dt.date.between(
                    pd.Timestamp(start_date).date(), pd.Timestamp(end_date).date()
                )
            ]
        )
        .join(procedures, "sagid", "inner")
        .dropna(subset="dato")
    )


def get_roll_calls(
    procedures,
    file_path=os.path.join(os.getcwd(), "Parliament CSV Files"),
    exclude_erroneous_roll_calls=True,
):
    return (
        pd.merge(
            left=pd.read_csv(
                os.path.join(file_path, "Afstemning.csv"),
                usecols=["sagstrinid", "id", "kommentar", "typeid"],
                index_col="id",
            ),
            right=procedures,
            left_on="sagstrinid",
            right_index=True,
        )
        .rename(columns={"id": "afstemningid"})
        .loc[
            lambda df: (
                df["typeid"] == 1
            )  # Exclude first and second readings of a bill
        ]
        .pipe(
            lambda df: (
                df.query("kommentar.isna()") if exclude_erroneous_roll_calls else df
            )
        )[["sagid", "dato"]]
    )


def get_votes(roll_calls, file_path=os.path.join(os.getcwd(), "Parliament CSV Files")):
    return (
        pd.merge(
            left=pd.read_csv(
                os.path.join(file_path, "Stemme.csv"),
                usecols=["afstemningid", "akt?rid", "typeid"],
            ),
            right=roll_calls,
            left_on="afstemningid",
            right_index=True,
        )
        .rename(columns={"akt?rid": "aktørid"})
        .replace({"typeid": {1: 5, 2: 1, 3: 0, 4: 3}})[
            ["typeid", "afstemningid", "aktørid"]
        ]
    )


def fix_corrupted_names(correct_names: pd.DataFrame, corrupted_names: pd.DataFrame):
    """
    Danish letters were lost when transfering the database. Uncorrupted names from a secondary
    source are used to restore the lost letters by matching the names to each other.
    Args:
        correct_names (pd.DataFrame): Dataframe column containing uncorrupted names ie. no missing letters.
        corrupted_names (pd.DataFrame): Dataframe column containing names with '?' instead of Danish letters.

    Returns:
        pd.Dataframe: The formerly corrupted names now correctly restored.
    """

    replacements = {"æ": "?", "Æ": "?", "ø": "?", "Ø": "?", "å": "?", "Å": "?"}

    replace_danish_letters = lambda name: "".join(
        replacements.get(letter, letter) for letter in name
    )

    names_with_danish_letters = (
        correct_names[["Candidates.Fullname"]]
        .assign(
            **{
                "Candidates.Fullname_normalised": lambda df: df[
                    "Candidates.Fullname"
                ].apply(replace_danish_letters)
            }
        )
        .drop_duplicates()
    )

    return corrupted_names.merge(
        right=names_with_danish_letters,
        how="inner",
        left_on="navn",
        right_on="Candidates.Fullname_normalised",
    )[["aktørid", "Candidates.Fullname"]]


def get_politicians(
    candidates: pd.DataFrame,
    file_path=os.path.join(os.getcwd(), "Parliament CSV Files"),
):
    """Gets the agents who appear in both datasets."""
    politicians = fix_corrupted_names(
        correct_names=candidates,
        corrupted_names=(
            pd.read_csv(os.path.join(file_path, "Aktør.csv"))
            .rename(columns={"id": "aktørid"})
            .loc[:, ["aktørid", "navn"]]
        ),
    )
    candidates = pd.merge(
        candidates,
        politicians[["Candidates.Fullname", "aktørid"]],
        on="Candidates.Fullname",
    )

    return politicians, candidates


def process_candidates(year: str):

    excel_file_prefix = "ChristianIverAndersen_Kandidattestdata.xlsx"
    candidate_test_data_file_path = os.path.join(os.getcwd(), "Candidate Test Files")

    ct_df: pd.DataFrame = pd.read_excel(  # ct_df = candidate test dataframe
        os.path.join(candidate_test_data_file_path, excel_file_prefix), sheet_name=year
    )

    ct_df = ct_df.rename(
        columns={
            ct_df.columns[1]: "Candidates.Firstname",
            ct_df.columns[2]: "Candidates.Lastname",
            ct_df.columns[3]: "Candidates.Party",
            ct_df.columns[7]: "Question",
            ct_df.columns[8]: "Answer",
        }
    ).dropna(subset=["Candidates.Firstname", "Candidates.Lastname"])
    ct_df["Candidates.Fullname"] = (
        ct_df["Candidates.Firstname"] + " " + ct_df["Candidates.Lastname"]
    )

    ct_df = ct_df[["Candidates.Fullname", "Candidates.Party", "Question", "Answer"]]

    if year == "FV19":
        ct_df["Answer"] = ct_df["Answer"].replace(
            {3: 4, 4: 5}
        )  # Aligns the scale with the other years

    return ct_df


def prune_node_outliers(
    votes: pd.DataFrame, id_column: str = "aktørid", std_dev: int = 2
):
    vote_counts = votes[id_column].value_counts()
    threshold = vote_counts.mean() - std_dev * vote_counts.std()
    reduced_votes = votes[
        votes[id_column].isin(vote_counts[vote_counts >= threshold].index)
    ]

    print(
        f"Nodes before: {votes[id_column].nunique()}\n"
        f"Nodes after: {reduced_votes[id_column].nunique()}\n"
        f"Nodes pruned: {(votes[id_column].nunique())-(reduced_votes[id_column].nunique())}"
    )
    return reduced_votes


def get_intersecting_agent_ids(
    pol: pd.DataFrame, cand: pd.DataFrame, votes: pd.DataFrame, id_to_party_map: dict
):

    votes = prune_node_outliers(votes[votes["aktørid"].isin(cand["aktørid"])])
    pol = pol[pol["aktørid"].isin(votes["aktørid"])]
    cand = cand[cand["aktørid"].isin(pol["aktørid"])]

    id_to_party_map = {
        agent_id: id_to_party_map[agent_id] for agent_id in cand["aktørid"]
    }

    if not (set(pol["aktørid"]) == set(cand["aktørid"]) == set(votes["aktørid"])):
        print(
            "Warning: IDs do not match between columns. Double-double check dataframes!"
        )

    return pol, cand, votes, id_to_party_map


def get_id_to_party_map(candidates: pd.DataFrame) -> dict:
    return candidates.set_index("aktørid").to_dict()["Candidates.Party"]


def enforce_party_discipline(
    votes: pd.DataFrame,
    id_to_party_map: dict,
    vote_type="typeid",
    roll_call="afstemningid",
    party="parti",
):
    whipped_votes = votes.copy()
    whipped_votes[party] = whipped_votes["aktørid"].map(id_to_party_map)
    ABSENT_VOTE = 0

    # Create a mapping dataframe for party-based replacements
    present_votes = (
        whipped_votes[whipped_votes[vote_type] != ABSENT_VOTE]
        .groupby([roll_call, party])[vote_type]
        .first()
        .reset_index()
        .rename(columns={vote_type: "partistemme"})
    )

    # Merge the non-zero votes back to the original dataframe
    whipped_votes = whipped_votes.merge(
        present_votes, on=[roll_call, party], how="left"
    )

    # Replace zero votes with party votes where applicable
    absent_mask = whipped_votes[vote_type] == ABSENT_VOTE
    whipped_votes.loc[absent_mask, vote_type] = whipped_votes.loc[
        absent_mask, "partistemme"
    ]

    # 3 is the value for abstentions, used as a last failsafe
    whipped_votes = whipped_votes.drop(columns=["partistemme"], errors="ignore").fillna(
        3
    )

    return whipped_votes


def pipeline_raw_data_to_dataframes(
    year: str, government_dates: dict, exclude_erroneous_roll_calls=True
):
    if year not in government_dates.keys():
        raise ValueError(
            f"Error! Please use a valid election year from {government_dates.keys()}."
        )

    politicians, candidates = get_politicians(process_candidates(year))
    id_to_party_map = get_id_to_party_map(candidates)
    votes = (
        get_procedures(year, government_dates)
        .pipe(get_roll_calls, exclude_erroneous_roll_calls=exclude_erroneous_roll_calls)
        .pipe(get_votes)
        .pipe(enforce_party_discipline, id_to_party_map)
    )
    return get_intersecting_agent_ids(politicians, candidates, votes, id_to_party_map)
