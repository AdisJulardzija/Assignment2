# Import necessary libraries
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy import stats
 
# Function to load the dataset
path = "Data/micro_world_139countries.csv"
df = pd.read_csv(path, encoding='latin-1') #Imports the FINDEX dataset from google drive, with latin-1 encoding
# df.columns #Shows the columns of the dataset
 
# The new dataframe ndf
ndf = df[["age", "fin44b", "fin44c"]] #We pick out the columns that we want to work with from the datset
# ndf.columns #We check the columns of the new dataframe
 
# Remove zero values and the 4, 5 and 6 responses
ndf = ndf.dropna() #We remove all rows with nulls
ndf = ndf[(ndf["fin44b"] < 4) & (ndf["fin44c"] < 4)] #We remove all rows where the respondent answered 4, 5 or 6 in either fin44c or fin44b
zage = zscore(ndf["age"]) #Finds the z-scores for the age column
ndf["outlier_age"] = (zage > 3) | (zage < -3) #Makes a boolean array that indicates whether a column has an outlier in age or not
ndf_cage = ndf[~ndf["outlier_age"]] #Removes all rows with outliers regarding age
ndf = ndf_cage[["age", "fin44b", "fin44c"]] #Removes the last column, as we have no use for them anymore
 
### The app
 
# Create age groups and add as a new column
bin_edges = [15, 30, 45, 60, 75, 100]
bin_labels = ['15-29', '30-44', '45-59', '60-74', '75+']
ndf['AgeGroup'] = pd.cut(ndf['age'], bins=bin_edges, labels=bin_labels, right=False)
 
# Display the DataFrame and its columns
#st.write("DataFrame with Age Groups:")
# st.dataframe(ndf)
 
#st.write("Columns in the DataFrame:")
# st.write(ndf.columns)
 
### The assignment begins here:
 
# Introduction
 
# Set the app title and sidebar header
st.title("Respondents' answers regarding their levels of worry about medical costs and bills")
st.sidebar.header("Filters")
 
# Objective of the site
with st.expander("**Objective**"):
                 st.markdown("""At the core of this dashboard is the goal of visually interpreting data to provide valuable insights on two critical aspects:
- Worriedness About future medical costs: How do different age groups perceive their future medical expenses, and which groups are more likely to be concerned about these costs?
- Worriedness About future bills: How do different age groups perceive their future medical expenses, and which groups are more likely to be concerned about these costs?
""")
 
# Sidebar filter: Age Group
selected_age_group = st.sidebar.multiselect("Select Age Groups", ndf['AgeGroup'].unique().tolist(), default=ndf['AgeGroup'].unique().tolist())
if not selected_age_group:
    st.warning("Please select an age group from the sidebar ⚠️")
    st.stop()
filtered_ndf = ndf[ndf['AgeGroup'].isin(selected_age_group)]
 
# Sidebar filter: level of wooriedness
medical_bills = ndf['fin44b'].unique().tolist()
selected_department = st.sidebar.multiselect("Select Level of worriedness", medical_bills, default=medical_bills)
if not selected_department:
    st.warning("Please select a level from the sidebar ⚠️")
    st.stop()
filtered_ndf = filtered_ndf[filtered_ndf['fin44b'].isin(medical_bills)]
 
# Sidebar filter: Age
min_age = int(ndf['age'].min())
max_age = int(ndf['age'].max())
Age = st.sidebar.slider("Select age", min_age, max_age)
 
# Displaying the legend for results
st.header("Results")
 
# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    " ",
    ["Medical costs",
     "Bills",
     "Box Plot of age of respondents",
     "Distribution of age groups"])
 
# Visualizations based on user selection
if visualization_option == "Medical costs":
    # Plotting the worriedness for medical costs across ages
    fig, ax = plt.subplots()
    sns.countplot(x="fin44b", data=ndf, ax=ax)
    ax.set_title("Distribution of Respondent's Worriedness of Medical Costs")
    st.pyplot(fig)
 
elif visualization_option == "Bills":
    # Plotting the worriedness for medical bills across ages
    fig, ax = plt.subplots()
    sns.countplot(x="fin44c", data=ndf, ax=ax)
    ax.set_title("Distribution of Respondent's Worriedness of Medical Bills")
    st.pyplot(fig)  
 
elif visualization_option == "Box Plot of age of respondents":
    # Plotting the Box Plot of age of respondents
    fig, ax = plt.subplots()
    sns.boxplot(x=ndf["age"], ax=ax)
    ax.set_title("Box Plot of Age of Respondents")
    st.pyplot(fig)  
 
elif visualization_option == "Distribution of age groups":
    #agechart
    chart = alt.Chart(ndf).mark_bar().encode(
        x=alt.X('AgeGroup', axis=alt.Axis(title='Age group')), y=alt.Y('count()', axis=alt.Axis(title='Amount')), color='fin44b').properties(title='Age groups')
    st.altair_chart(chart, use_container_width=True)
 
# Displaying dataset overview
st.header("Dataset Overview")
st.dataframe(ndf.describe())
 
# Displaying the Attrition Analysis header
st.header("Correlation")
    
corrmat = ndf[["age", "fin44b", "fin44c"]].corr(method="spearman") #We create a correlation matrix, to look at the correlations between the different variables
fig, ax = plt.subplots()
sns.heatmap(corrmat, annot=True, cmap="coolwarm").set_title("Correlation Matrix") #We create a heatmap, so we can visually see the correlations between each variable.
st.pyplot(fig)
 
# Adding interpretation text
st.markdown("""After looking at the correlations between the variables,
            we can also look at whether there is any difference in age,
            in regards to how worried the respondents are. We perform an ANOVA test,
            to test the hypothesis. The H0-hypothesis is that there is no significant difference in age,
            for each level of worriedness in relation to medical costs.
""")
 
# Displaying dataset overview
st.header("ANOVA test results")
 
# Performing ANOVA test
level_worried1 = ndf[ndf["fin44b"] == 1]["age"]  # We filter the worriedness of medical costs to only include people with very high worriedness
level_worried2 = ndf[ndf["fin44b"] == 2]["age"]  # We filter the worriedness of medical costs to only include people with medium worriedness
level_worried3 = ndf[ndf["fin44b"] == 3]["age"]  # We filter the worriedness of medical costs to only include people with no worriedness
 
f_stat, p_value = stats.f_oneway(level_worried1, level_worried2, level_worried3)
 
# Displaying ANOVA test results
st.write(f"F-statistic: {f_stat}")
st.write(f"P-value: {p_value}")
 
# Creating and displaying boxplot for age distribution by worriedness levels on medical costs
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="fin44b", y="age", data=ndf, ax=ax)
ax.set_title("Age Distribution by Worriedness of Medical Costs")
ax.set_xlabel("Worriedness Level (fin44b)")
ax.set_ylabel("Age")
st.pyplot(fig)
 
# Displaying dataset overview
st.header("Conclusion")
 
# Adding interpretation conclusion
with st.expander("What have we learned of the link between age and worriedess of medical costs and bills?"):
    st.markdown("""
    1. **Age and worriedness** - The correlation matrix and ANOVA test suggests that people tend to worry less about medical costs and bills as they age.
    2. **Correlation strength** - We saw that the correlation betweeen age and worriedness seem to be low, which suggests a weak link between the two.
    3. **Distribution of respondents** - The distribution of respondents lean pretty heavily towards younger respondents, which might mean that older people are not represented correctly.
    4. **Distribution of worriedness** - The distribution of worriedness seems to suggest that generally people are worried about medical costs, but not necessarily bills.
                Although a higher representation of worried people might be explained the the higher representation of younger people, who seemingly worry more.
                """)
