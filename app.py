import streamlit as st
from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    last_names,
    visualize_families,
)

st.header("Exercise 1: Survival Patterns")

st.write("**Question:** Did adult women in first class have the highest survival rate?")

demo_table = survival_demographics()
st.dataframe(demo_table)
st.plotly_chart(visualize_demographic())


st.header("Exercise 2: Family Size and Wealth")

st.write("**Question:** Do larger families in first class consistently pay higher fares?")

family_table = family_groups()
st.dataframe(family_table)

st.write("### Last Name Counts")
st.write(last_names())

st.plotly_chart(visualize_families())