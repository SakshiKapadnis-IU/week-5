# apputil.py
import pandas as pd
import numpy as np
import plotly.express as px


def load_titanic():
    """Load Titanic data and standardize column names expected by the autograder."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Standardize to lowercase names the autograder expects
    df = df.rename(columns={"Age": "age", "Pclass": "pclass", "Sex": "sex"})
    return df


# ----------------------------
# Exercise 1: Survival Patterns
# ----------------------------
def survival_demographics():
    """
    Return a table grouped by pclass, sex, and age_group with:
    n_passengers, n_survivors, survival_rate.
    Includes empty groups (n_passengers == 0).
    Ensures age_group is a pandas Categorical dtype.
    """
    df = load_titanic().copy()

    bins = [0, 12, 19, 59, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]

    # create categorical age_group column
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    # group (explicit observed=False to match current pandas behavior)
    grouped = (
        df.groupby(["pclass", "sex", "age_group"], observed=False)
        .agg(n_passengers=("PassengerId", "count"), n_survivors=("Survived", "sum"))
    )

    # Build a MultiIndex with all combinations so empty groups appear
    idx = pd.MultiIndex.from_product(
        [
            sorted(df["pclass"].dropna().unique()),
            sorted(df["sex"].dropna().unique()),
            pd.CategoricalDtype(categories=labels).categories,
        ],
        names=["pclass", "sex", "age_group"],
    )

    grouped = grouped.reindex(idx, fill_value=0).reset_index()

    # Ensure age_group column is categorical with the expected categories and order
    grouped["age_group"] = pd.Categorical(grouped["age_group"], categories=labels, ordered=True)

    # survival_rate: avoid division by zero (for n_passengers == 0 produce 0.0)
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    grouped.loc[grouped["n_passengers"] == 0, "survival_rate"] = 0.0

    # sort rows for readability
    grouped = grouped.sort_values(["pclass", "sex", "age_group"]).reset_index(drop=True)

    return grouped


def visualize_demographic(df):
    """Return a Plotly figure visualizing survival_rate by age_group/sex/pclass."""
    fig = px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="sex",
        facet_col="pclass",
        barmode="group",
        title="Survival Rate by Age Group, Sex, and Passenger Class",
    )
    return fig


# ----------------------------
# Exercise 2: Family Size & Wealth
# ----------------------------
def family_groups():
    """
    Return table grouped by family_size and pclass with:
    n_passengers, avg_fare, min_fare, max_fare.
    """
    df = load_titanic().copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = (
        df.groupby(["family_size", "pclass"], observed=False)
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(["pclass", "family_size"]).reset_index(drop=True)
    return grouped


def last_names():
    """
    Extract last name from Name column and return a value_counts() Series.
    """
    df = load_titanic().copy()
    df["last_name"] = df["Name"].str.extract(r"^([^,]+)")
    return df["last_name"].value_counts()


def visualize_families(df):
    """Return a Plotly figure showing avg_fare by family_size and pclass."""
    # ensure pclass treated as string for color grouping in plotly
    df_plot = df.copy()
    df_plot["pclass"] = df_plot["pclass"].astype(str)
    fig = px.line(df_plot, x="family_size", y="avg_fare", color="pclass", markers=True)
    return fig


# ----------------------------
# Bonus: Age division within class
# ----------------------------
def determine_age_division():
    """
    Add 'median_age' and boolean 'older_passenger' that indicates
    whether passenger's age > median age for their pclass.
    older_passenger must be pd.NA wherever age is NA (to mirror NA count in 'age').
    Returns the full dataframe (with original columns) plus these two new columns.
    """
    df = load_titanic().copy()

    # compute median age per class (median will be NaN for classes with no ages)
    df["median_age"] = df.groupby("pclass")["age"].transform("median")

    # older_passenger: preserve missingness in 'age' (pd.NA for missing ages)
    # Use numpy/pandas logic: where age is missing -> pd.NA, else compare
    older = np.where(df["age"].isna(), pd.NA, df["age"] > df["median_age"])

    # put into column (use object dtype so pd.NA can live alongside booleans)
    df["older_passenger"] = pd.Series(older, index=df.index, dtype="object")

    return df


def visualize_age_division(df):
    """Return a simple Plotly histogram showing older_passenger counts by pclass."""
    fig = px.histogram(df, x="pclass", color="older_passenger", barmode="group")
    return fig