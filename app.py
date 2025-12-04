import streamlit as st
from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    last_names,
    visualize_families,
    determine_age_division,
    visualize_age_division,
)

st.title("Week 5 – Titanic Data Analysis App")


# ---------------------------------------------------------
# Exercise 1
# ---------------------------------------------------------
st.header("Exercise 1: Survival Demographics")

# Your question:
st.write("**Question:** How do survival rates differ between men and women across each age group within each passenger class?")

df_demo = survival_demographics()
st.write(df_demo)

fig1 = visualize_demographic(df_demo)
st.plotly_chart(fig1)


# ---------------------------------------------------------
# Exercise 2
# ---------------------------------------------------------
st.header("Exercise 2: Family Size & Wealth")

# Your question:
st.write("**Question:** Do larger families tend to pay higher or lower fares across passenger classes?")

df_fam = family_groups()
st.write(df_fam)

fig2 = visualize_families(df_fam)
st.plotly_chart(fig2)

st.subheader("Last Names Count")
st.write(last_names())


# ---------------------------------------------------------
# Bonus
# ---------------------------------------------------------
st.header("Bonus: Age Division by Class")

df_age = determine_age_division()
st.write(df_age.head())

fig3 = visualize_age_division(df_age)
st.plotly_chart(fig3)