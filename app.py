import numpy as np
import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Diabetes Regression", page_icon=":hospital:", layout="wide")



####################################################################################################################################

# ---- READ EXCEL ----
@st.cache
def get_data_from_excel():
    df = pd.read_excel(
        io="supermarkt_sales.xlsx",
        engine="openpyxl",
        sheet_name="Sales",
        skiprows=3,
        usecols="B:R",
        nrows=1000,
    )
    # Add 'hour' column to dataframe
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    return df

df = get_data_from_excel()

#Add another dataframe
diabetes = datasets.load_diabetes() # load data

# Sperate train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
# There are three steps to model something with sklearn
# 1. Set up the model
model = LinearRegression()
# 2. Use fit
model.fit(X_train, y_train)
# 3. Check the score
Model_Score = model.score(X_test, y_test)

####################################################################################################################################


# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")
city = st.sidebar.multiselect(
    "Select the City:",
    options=df["City"].unique(),
    default=df["City"].unique()
)

customer_type = st.sidebar.multiselect(
    "Select the Customer Type:",
    options=df["Customer_type"].unique(),
    default=df["Customer_type"].unique(),
)

gender = st.sidebar.multiselect(
    "Select the Gender:",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

df_selection = df.query(
    "City == @city & Customer_type ==@customer_type & Gender == @gender"
)

####################################################################################################################################


# ---- MAINPAGE ----
st.title(":hospital: Diabetes Regression Model")
st.markdown("##")

# TOP KPI's
#total_sales = int(df_selection["Total"].sum())
Model_Score= (float(Model_Score ))*100

def pred_index(number):
    Prediction = model.predict(X_test[int(number):int(number+1),:])
    return Prediction[0]

average_rating = round(df_selection["Rating"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Model Score is : ")
    st.subheader(f"{'{:.2f}'.format(Model_Score)} %")
with middle_column:
    number = st.number_input('Insert an index number of X_test to predict')
    Predict = pred_index(number)
    st.write('The perdicted value is ', Predict)
#with middle_column:
#    st.subheader("Average Rating:")
#    st.subheader(f"{average_rating} {star_rating}")
with right_column:
    st.subheader("Average Sales Per Transaction:")
    st.subheader(f"US $ {average_sale_by_transaction}")

st.markdown("""---""")



####################################################################################################################################

y_pred = model.predict(X_test) 

# Linear Regression Graph
#df = px.data.tips()
#fig = px.scatter(df, x="total_bill", y="tip", trendline="ols")
#fig.show()

preds_df =  pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
#preds_df = px.data.tips()

fig_reg = px.scatter(
    preds_df,
    x="actual",
    y="predictions",
    trendline="ols",
    title="<b>Linear Regression Graph</b>",
    #color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
    template="plotly_white",
)
fig_reg.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

# SALES BY HOUR [BAR CHART]
sales_by_hour = df_selection.groupby(by=["hour"]).sum()[["Total"]]
fig_hourly_sales = px.bar(
    sales_by_hour,
    x=sales_by_hour.index,
    y="Total",
    title="<b>Sales by hour</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
    template="plotly_white",
)
fig_hourly_sales.update_layout(
    xaxis=dict(tickmode="linear"),
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(showgrid=False)),
)


left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
right_column.plotly_chart(fig_reg, use_container_width=True)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# Resources
# @Email:  contact@pythonandvba.com
# @Website:  https://pythonandvba.com
# @YouTube:  https://youtube.com/c/CodingIsFun
# @Project:  Sales Dashboard w/ Streamlit
# Model Source : https://www.kaggle.com/andyxie/beginner-scikit-learn-linear-regression-tutorial
# regression plot : https://plotly.com/python/linear-fits/