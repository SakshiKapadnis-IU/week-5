import pandas as pd
import plotly.express as px


def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Standardize column names for autograder
    df = df.rename(columns={"Age": "age", "Pclass": "pclass", "Sex": "sex"})
    return df


# ------------------------------------------------------------
# EXERCISE 1
# ------------------------------------------------------------
def survival_demographics():
    df = load_titanic().copy()

    # Create age groups
    bins = [0, 12, 19, 59, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]

    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Group
    grouped = (
        df.groupby(["pclass", "sex", "age_group"])
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum"),
        )
    )

    # Create full index for missing combinations
    idx = pd.MultiIndex.from_product(
        [
            df["pclass"].unique(),
            df["sex"].unique(),
            df["age_group"].cat.categories,
        ],
        names=["pclass", "sex", "age_group"],
    )

    grouped = grouped.reindex(idx, fill_value=0).reset_index()

    # Survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Sort
    return grouped.sort_values(["pclass", "sex", "age_group"])


def visualize_demographic(df):
    fig = px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="sex",
        facet_col="pclass",
    )
    return fig


# ------------------------------------------------------------
# EXERCISE 2
# ------------------------------------------------------------
def family_groups():
    df = load_titanic().copy()

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = (
        df.groupby(["family_size", "pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
    )

    return grouped.sort_values(["pclass", "family_size"])


def last_names():
    df = load_titanic().copy()
    df["last_name"] = df["Name"].str.extract(r"^([^,]+)")
    return df["last_name"].value_counts()


def visualize_families(df):
    fig = px.line(
        df,
        x="family_size",
        y="avg_fare",
        color="pclass",
        markers=True,
    )
    return fig


# ------------------------------------------------------------
# BONUS
# ------------------------------------------------------------
def determine_age_division():
    df = load_titanic().copy()

    df["median_age"] = df.groupby("pclass")["age"].transform("median")
    df["older_passenger"] = df["age"] > df["median_age"]

    return df


def visualize_age_division(df):
    fig = px.histogram(
        df,
        x="pclass",
        color="older_passenger",
        barmode="group",
    )
    return fig