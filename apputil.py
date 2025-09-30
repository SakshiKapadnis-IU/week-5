import pandas as pd
import plotly.express as px

def survival_demographics(df):
    bins = [0, 12, 19, 59, 120]
    labels = ["Child", "Teen", "Adult", "Senior"]
    age_cat = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    df["age_group"] = pd.Categorical(age_cat, categories=labels, ordered=True)

    grouped = df.groupby(["Pclass", "Sex", "age_group"]).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum"),
    )

    all_pclasses = [1, 2, 3]
    all_sexes = ["female", "male"]
    all_combinations = pd.MultiIndex.from_product(
        [all_pclasses, all_sexes, labels],
        names=["Pclass", "Sex", "age_group"],
    )

    grouped = grouped.reindex(all_combinations, fill_value=0).reset_index()
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    grouped.loc[grouped["n_passengers"] == 0, "survival_rate"] = 0.0
    grouped["age_group"] = pd.Categorical(grouped["age_group"], categories=labels, ordered=True)
    grouped = grouped.sort_values(["Pclass", "Sex", "age_group"]).reset_index(drop=True)
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
        labels={"survival_rate": "Survival Rate", "age_group": "Age Group"},
        category_orders={"age_group": ["Child", "Teen", "Adult", "Senior"]}
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(yaxis_tickformat='.0%', yaxis_title="Survival Rate", xaxis_title="Age Group")
    return fig
