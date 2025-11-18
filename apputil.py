import pandas as pd
import plotly.express as px
from load_data import load_data

def survival_demographics():
    df = load_data()

    bins = [0, 12, 19, 59, 200]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    df["age_group"] = df["age_group"].astype("category")

    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum")
        )
        .reset_index()
    )
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    return grouped.sort_values(["Pclass", "Sex", "age_group"])


def visualize_demographic():
    table = survival_demographics()
    fig = px.bar(
        table,
        x="age_group",
        y="survival_rate",
        color="Sex",
        barmode="group",
        facet_col="Pclass",
        labels={"survival_rate": "Survival Rate"},
        title="Survival Rate by Class, Sex, and Age Group"
    )
    return fig


def family_groups():
    df = load_data()

    df["family_size"] = df["SibSp"] + df["Parch"] + 1

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
    return grouped.sort_values(["Pclass", "family_size"])


def last_names():
    df = load_data()

    df["last_name"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
    return df["last_name"].value_counts()


def visualize_families():
    table = family_groups()
    fig = px.line(
        table,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        title="Average Fare by Family Size and Passenger Class"
    )
    return fig