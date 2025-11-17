import streamlit as st
from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    last_names,
    visualize_families,
)

st.header("Exercise 1: Survival Patterns")

# Produce table
demo_table = survival_demographics(df)

# 6. Your question for Exercise 1
st.write("**Question:** Did adult women in first class have the highest survival rate among all groups?")

# Show table
st.dataframe(demo_table)

# Plot
fig_demo = visualize_demographic(demo_table)
st.plotly_chart(fig_demo)


st.header("Exercise 2: Family Size and Wealth")

family_table = family_groups(df)

# Your question for Exercise 2
st.write("**Question:** Do larger families in first class consistently pay higher fares on average?")

# Show table
st.dataframe(family_table)

# Last names result
name_counts = last_names(df)
st.write("### Last Name Counts (Do they match family-size patterns?)")
st.write(name_counts)

# Plot
fig_family = visualize_families(family_table)
st.plotly_chart(fig_family)