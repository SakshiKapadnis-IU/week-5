import pandas as pd
import plotly.express as px


# ---------------------------------------------------------
# Load Titanic dataset
# ---------------------------------------------------------
def load_titanic():
    """Load Titanic dataset."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)


# ---------------------------------------------------------
# Exercise 1 — Survival Patterns
# ---------------------------------------------------------
def survival_demographics():
    """
    Create age groups, group by class/sex/age group,
    and calculate survival statistics.
    """
    df = load_titanic().copy()

    # Create age category column
    bins = [0, 12, 19, 59, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]

    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    # Group
    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum"),
        )
        .reset_index()
    )

    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Sort for readability
    grouped = grouped.sort_values(["Pclass", "Sex", "age_group"])

    return grouped


def visualize_demographic(df):
    """
    Create a Plotly visualization based on your question.
    Here: Compare survival rate across sex & age groups by class.
    """
    fig = px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        title="Survival Rates by Age Group, Gender, and Passenger Class",
    )
    return fig


# ---------------------------------------------------------
# Exercise 2 — Family Size & Wealth
# ---------------------------------------------------------
def family_groups():
    """Group by family size & class, compute fare stats."""
    df = load_titanic().copy()

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    grouped = (
        df.groupby(["family_size", "Pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(["Pclass", "family_size"])
    return grouped


def last_names():
    """Extract last names and count occurrences."""
    df = load_titanic().copy()

    df["last_name"] = df["Name"].str.extract(r"^([^,]+)")
    return df["last_name"].value_counts()


def visualize_families(df):
    """
    Example visualization:
    Average fare by family size across classes.
    """
    fig = px.line(
        df,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        title="Average Fare by Family Size and Passenger Class",
    )
    return fig


# ---------------------------------------------------------
# Bonus — Age Division Within Class
# ---------------------------------------------------------
def determine_age_division():
    """Create a column marking whether passenger is older than class median age."""
    df = load_titanic().copy()

    df["median_class_age"] = df.groupby("Pclass")["Age"].transform("median")
    df["older_passenger"] = df["Age"] > df["median_class_age"]

    return df


def visualize_age_division(df):
    """
    Visualize proportion of older passengers by class & survival.
    """
    fig = px.histogram(
        df,
        x="Pclass",
        color="older_passenger",
        barmode="group",
        title="Older vs Younger Passengers by Class",
    )
    return fig