import pandas as pd
import plotly.express as px


def survival_demographics(df):
    """Analyze survival by class, sex, and age groups."""

    # 1. Create age groups
    bins = [0, 12, 19, 59, 150]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    # 2. Group
    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum")
        )
        .reset_index()
    )

    # 3. Add survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # 4. Sort results
    grouped = grouped.sort_values(["Pclass", "Sex", "age_group"])

    return grouped


def visualize_demographic(table):
    """Return a Plotly chart that visualizes demographic survival rates."""
    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        title="Survival Rate by Class, Sex, and Age Group",
        labels={"survival_rate": "Survival Rate"}
    )
    fig.update_layout(height=500)
    return fig


def family_groups(df):
    """Analyze family size vs. fare and class."""

    # 1. Create family size variable
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # 2. Group
    grouped = (
        df.groupby(["family_size", "Pclass"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max")
        )
        .reset_index()
    )

    # 3. Sort clearly
    grouped = grouped.sort_values(["Pclass", "family_size"])

    return grouped


def last_names(df):
    """Extract last names and count frequency."""

    # Names are formatted: "LastName, Title Firstname..."
    df["last_name"] = df["Name"].apply(lambda x: x.split(",")[0].strip())

    counts = df["last_name"].value_counts()
    return counts


def visualize_families(table):
    """Plot average fare by family size and class."""

    fig = px.line(
        table,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        title="Average Fare by Family Size and Passenger Class"
    )
    fig.update_layout(height=500)
    return fig