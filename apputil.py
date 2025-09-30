import pandas as pd
import plotly.express as px

def survival_demographics(df):
    bins = [0, 12, 19, 59, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    grouped = (
        df.groupby(["Pclass", "Sex", "age_group"])
        .agg(
            n_passengers=("PassengerId", "count"),
            n_survivors=("Survived", "sum"),
        )
    )

    all_combinations = pd.MultiIndex.from_product(
        [df["Pclass"].unique(), df["Sex"].unique(), labels],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = grouped.reindex(all_combinations, fill_value=0).reset_index()
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    grouped.loc[grouped["n_passengers"] == 0, "survival_rate"] = 0

    grouped["age_group"] = pd.Categorical(grouped["age_group"], categories=labels, ordered=True)
    return grouped

def visualize_demographic(df):
    fig = px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        text="survival_rate",
        title="Survival Rate by Class, Sex, and Age Group",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[0, 1])
    return fig

def family_groups(df):
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    grouped = (
        df.groupby(["Pclass", "family_size"])
        .agg(
            n_passengers=("PassengerId", "count"),
            avg_fare=("Fare", "mean"),
            min_fare=("Fare", "min"),
            max_fare=("Fare", "max"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["Pclass", "family_size"]).reset_index(drop=True)
    grouped = grouped[["Pclass", "family_size", "n_passengers", "avg_fare", "min_fare", "max_fare"]]
    return grouped

def last_names(df):
    df["last_name"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
    last_name_counts = df["last_name"].value_counts().sort_index()
    return last_name_counts

def visualize_families(df):
    fig = px.scatter(
        df,
        x="family_size",
        y="avg_fare",
        size="n_passengers",
        color="Pclass",
        hover_data=["min_fare", "max_fare"],
        title="Average Fare by Family Size and Class",
    )
    return fig
