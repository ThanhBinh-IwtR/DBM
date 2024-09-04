import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io
from final import maindata


df = maindata()

# Configure the page
st.set_page_config(
    page_title="Exploratory Data Analysis of Greenhouse Gas Giants Dataset",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to initialize session state
def initialize_state(state_name, default_value=False):
    if state_name not in st.session_state:
        st.session_state[state_name] = default_value

# Initialize session state for main menu and submenus
main_menu = ["Data Information", "EDA", "Machine Learning", "Data Dashboard"]
data_info_options = ["Data Description",
                     "Data Discovery", 
                     "Data Info", 
                     "Unique Values", 
                     "Missing Values"]
eda_options = ["Production by Commodity",
               "Total Production Value by Year for Each Parent Entity",
               "Compare Total Production Value and Emissions over Years by Parent Type",
               "Average global Fossil Fuels and Mineral production and CO2 emission over years"]
mln_options = ["Data Preprocess",
               "Models",
               "Total Compare"]

model_options = ["Linear Regression",
                 "XG Boost",
                 "Lighboost",
                 "Random Forest",
                 "Catboost",
                 "Decision Tree Regression",
                 "Polynomial Regression",
                 "Support Vector Regressor"]

for option in main_menu:
    initialize_state(option)
for option in data_info_options:
    initialize_state(option)
for option in eda_options:
    initialize_state(option)
for option in mln_options:
    initialize_state(option)
for option in model_options:
    initialize_state(option)

# Sidebar main menu
st.sidebar.header("BÀI NÀY KHÔNG 10Đ THÌ CẢ NHÓM NGHỈ HỌC")
with st.sidebar:
  st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSt2WcvQotQdg9-M9oMOiPVpE6aVoYOxpNH5w&s", width=150)

st.sidebar.title("Main Menu")
selected_main_menu = st.sidebar.radio("Select a menu", main_menu)

# Show submenus based on main menu selection
if selected_main_menu == "":
    st.title("Welcome to our project")
if selected_main_menu == "Data Information":
    st.sidebar.title("Data Information Menu")
    for option in data_info_options:
        st.session_state[option] = st.sidebar.checkbox(option, value=st.session_state[option])
if selected_main_menu == "EDA":
    st.sidebar.title("Data EDA Menu")
    for option in eda_options:
        st.session_state[option] = st.sidebar.checkbox(option, value=st.session_state[option])
if selected_main_menu == "Machine Learning":
    st.sidebar.title("Machine Learning Menu")
    for option in mln_options:
        st.session_state[option] = st.sidebar.checkbox(option, value=st.session_state[option])

    st.sidebar.title("Model Menu")
    for option in model_options:
        st.session_state[option] = st.sidebar.checkbox(option, value=st.session_state[option])

# Load data
# data_path = "emissions_medium_granularity.csv"

raw = df
st.title("Exploratory Data Analysis of Greenhouse Gas Giants Dataset")

if selected_main_menu == "Data Information":
    if st.session_state["Data Description"]:
        st.header("Data Description")
        st.markdown("""
        **Dataset:** Greenhouse gas giants

        **Carbon Majors Data has the following features:**
        - **Open Source:** The data is available for download as CSV files for non-commercial use. InfluenceMap's Terms and Conditions apply.
        - **Annual Updates:** The data is updated annually in November, and the downloads represent the latest available data.

        **Levels of Data Granularity:**
        1. **Low Granularity:** Includes year, entity, entity type, and total emissions.
        2. **Medium Granularity:** Includes year, entity, entity type, commodity, commodity production, commodity unit, and total emissions.
        3. **High Granularity:** Includes the same fields as the medium granularity file, as well as the reporting entity, data point source, product emissions, and four different operational emissions: flaring, venting, own fuel use, and fugitive methane.

        => **Chosen File:** emissions_medium_granularity file
        """)

    if st.session_state["Data Discovery"]:
        st.header("Data Discovery")

        st.subheader("Common Data Information")
        st.write(f"Total raw data: {raw.shape}")
        st.write(raw.head())

        st.write("Dataset has 12551 examples and 7 features:")
        st.write("""
        - **year:** The year of the data point
        - **parent_entity:** The entity to whom the emissions are traced to (fuel distributor, fuel company,...)
        - **parent_type:** investor-owned company/state-owned entity/nation state.
        - **commodity**
        - **production_value:** The quantity of production
        - **production_unit:** The unit of production
        - **total_emissions_MtCO2e:** The total emissions
        """)

    if st.session_state["Data Info"]:
        st.header("Data Info")
        buffer = io.StringIO()
        raw.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    if st.session_state["Unique Values"]:
        st.header("Unique Values in Each Column")
        dataset_without_total_column = raw.drop(columns=["total_emissions_MtCO2e", "production_value"])
        for column in dataset_without_total_column.columns:
            unique_value = dataset_without_total_column[column].nunique()
            st.write(f"{column}: {unique_value} unique values")

        st.write("""
        Explain: The dataset contains information about various entities involved in the production of different commodities. 
        - There are 122 unique parent entities categorized into three types: State-owned Entity, Investor-owned Company, and Nation State. 
        - These entities produce nine different commodities, including Oil & NGL, Natural Gas, and various types of coal such as Sub-Bituminous Coal, Metallurgical Coal, and Bituminous Coal, as well as Cement and Lignite Coal. 
        - The production units used to measure these commodities are Million bbl/yr, Bcf/yr, Million tonnes/yr, and Million Tonnes CO2.
        """)

    if st.session_state["Missing Values"]:
        st.header("Missing Values")
        st.write("The dataset has no null values.")

elif selected_main_menu == "EDA":
    if st.session_state["Production by Commodity"]:
        st.header("Production by Commodity")
        # Calculate the total amount of production value of each commodity
        group_data_by_commodity = raw.groupby(raw['commodity'])['production_value'].sum().reset_index()
        st.text(group_data_by_commodity)
        st.write("""
        Natural gas and oil & NGL have significantly higher total production compared to other products, so using a logarithmic scale makes it easier to observe while maintaining the order of product production and changing the colors for better aesthetics.
        """
        )
        data_sorted = group_data_by_commodity.sort_values(by="production_value", ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='production_value', y='commodity', data=data_sorted, palette='viridis')

        plt.xscale('log')
        plt.xlabel('Production Value (log scale)')
        plt.ylabel('Commodity')
        plt.title('Production Value by Commodity')

        for index, value in enumerate(data_sorted['production_value']):
            plt.text(value, index, f'{value:.0f}', va='center')

        plt.tight_layout()
        st.pyplot(plt)

        st.write("""
            Explain: 
            - Natural Gas has the highest production value by far, amounting to approximately 3.56 million units. This suggests that Natural Gas is a critical commodity in terms of production value.
            - Oil & NGL (Natural Gas Liquids) also shows a high production value at approximately 1.32 million units, indicating its substantial role in the commodity market.
            - Among the different types of coal, Bituminous Coal has the highest production value at approximately 121,867 units. This highlights its significant contribution compared to other coal types.
            """)
    if st.session_state["Total Production Value by Year for Each Parent Entity"]:
        st.header("Total Production Value by Year for Each Parent Entity")
        group_data_by_year_and_parent_entity = raw.pivot_table(values = ["production_value", "total_emissions_MtCO2e"], index = ["year","parent_entity"], aggfunc = "sum").reset_index()
        st.text(group_data_by_year_and_parent_entity)
        fig = px.line(group_data_by_year_and_parent_entity, x='year', y='production_value', color='parent_entity', title='Total Production Value by Year for Each Parent Entity')
        st.plotly_chart(fig)
    if st.session_state["Compare Total Production Value and Emissions over Years by Parent Type"]:
        st.header("Compare Total Production Value and Emissions over Years by Parent Type")

        group_data_by_year_and_parent_entity = raw.pivot_table(values = ["production_value"], index = ["year","parent_type"], aggfunc = "sum").reset_index()
        st.text(group_data_by_year_and_parent_entity)

        fig = px.line(
                group_data_by_year_and_parent_entity, x='year', y='production_value',
                color='parent_type', line_group='parent_type',
                title='Total Production Value over Years by Parent Type',
                labels={'year': 'Year', 'production_value': 'Production Value', 'parent_type': 'Parent Type'}
            )
        st.plotly_chart(fig)

        group_data_by_year_and_parent_entity = raw.pivot_table(values=["total_emissions_MtCO2e"], 
                                                        index = ["year","parent_type"], 
                                                        aggfunc = "sum").reset_index()
        fig = px.line(group_data_by_year_and_parent_entity, x='year', y='total_emissions_MtCO2e',
                        color='parent_type', line_group='parent_type',
                        title='Total Emissions over Years by Parent Type',
                        labels={'year': 'Year', 'value': 'Value', 'parent_type': 'Parent Type', 'variable': 'Metric'})
        st.plotly_chart(fig)

        group_data_by_year_and_parent_entity = raw.pivot_table(values=['production_value', 'total_emissions_MtCO2e'], 
                                                        index = ["year","parent_type"], 
                                                        aggfunc = "sum").reset_index()
        fig = px.bar(group_data_by_year_and_parent_entity, x='parent_type', y=['production_value', 'total_emissions_MtCO2e'],
                    barmode='group', title='Comparison of Production Value and Total Emissions by Parent Type',
                    labels={'parent_type': 'Parent Type', 'value': 'Value', 'variable': 'Metric'})
        st.plotly_chart(fig)

        st.write("""
        Explain: 
        Most energy companies are state-controlled due to the importance of energy security. A nation typically needs to ensure three types of security: food security, national defense, and energy security to maintain overall sovereignty. Therefore, it is understandable that state-owned enterprises comprise more than 50% of the sector.
        """)

    if st.session_state["Average global Fossil Fuels and Mineral production and CO2 emission over years"]:
        st.header("Average Global Fossil Fuels and Mineral Production and CO2 Emission over Years")
        # Create pivot mean table
        data_pvt = raw.pivot_table(values = ["production_value", "total_emissions_MtCO2e"], index = "year", aggfunc = "mean").reset_index()
        st.write(data_pvt.head())

        data_melted = pd.melt(data_pvt, id_vars=['year'], value_vars=['production_value', 'total_emissions_MtCO2e'], var_name='variable', value_name='value')
        fig = px.line(data_melted, x='year', y='value', color='variable', title='Avg. global Fossil Fuels and Mineral production & CO2 emissions over the years')

        fig.update_layout(
                            xaxis_title='Year',
                            yaxis_title='Mean Value',
                            legend_title_text='Variable',
                            template='plotly_white',
                            title_x=0.5
                        )
        st.plotly_chart(fig)
        
        st.subheader("The correlation")
        st.write("""
        The total production of Fossil Fuels and Mineral is directly proportional to the amount of CO2 emissions, but the correlation between these two factors varies from year to year. Specifically, Fossil Fuels and Mineral production increases rapidly over time, while CO2 emissions increase but at a much slower pace. 
        There could be several different reasons:
        - Increased energy efficiency
        - Energy conversion and the use of alternative fuels
        - Carbon Capture and Storage (CCS)
        - Environmental policies and regulations
        - Structural changes in the energy industry
        """)
        st.subheader("The growth rate")
        st.write("""
        - The Fossil Fuels and Mineral production began to increase around the 1880s following the establishment of Edison's electric company, Edison Electric Light Company, in 1882.
        - During the oil crises of 1973-1974 and 1979-1980, the decrease in oil production and price hikes led to a significant decline in ONGL production in the 1970s. After a period of instability, production began to stabilize and increase again. During this time, CO2 emissions also decreased. In 1997, countries that signed the Kyoto Protocol committed to reducing greenhouse gas emissions. Additionally, advancements in machinery efficiency and the emergence of new technologies contributed to a significant reduction in CO2 emissions.
        - 2008-2009 Global Financial Crisis: The global financial crisis resulted in a decrease in energy demand and reduced production across various energy sources, including oil and natural gas. Following a slight decline, production then rebounded rapidly.
        """)

elif selected_main_menu == "Machine Learning":

    # df = pd.read_csv("emissions_medium_granularity.csv")
    df.drop('production_unit', axis=1, inplace=True)
    df[['parent_type', 'parent_entity','commodity']] = df[['parent_type', 'parent_entity','commodity']].apply(LabelEncoder().fit_transform)

    if st.session_state["Data Preprocess"]:
        st.header("Data Preprocess")
        st.write("Use with new_df set to predict total_emissions_MtCO2e")
        st.write(raw.head())
        
        st.subheader("Drop production_unit column")
        code = "df.drop('production_unit', axis=1, inplace=True)"
        st.code(code, language='python')

        st.subheader("Encoding Label")
        code = """
            le = LabelEncoder()
            df[['parent_type', 'parent_entity','commodity']] = df[['parent_type', 'parent_entity','commodity']].apply(LabelEncoder().fit_transform)
            """
        st.code(code, language='python')
        st.write(df.head())

        st.subheader("Divide the training and testing sets in the ratio 80:20")
        code = """
            X = df.drop(['total_emissions_MtCO2e'], axis=1)
            y = df['total_emissions_MtCO2e']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            """
        st.code(code, language='python')

    selected_models = [model for model in model_options if st.session_state[model]]
    if st.session_state["Models"]:
        if "Linear Regression" in selected_models:
            st.header("Linear Regression")
            code = """
            model = LinearRegression()
            scores = cross_val_score(model, X_train, y_train, cv=10,scoring ='r2')
            """
            st.code(code, language='python')
            linear_r2_scores = np.load('scores/linear_r2_scores.npy')
            st.write(f"R² scores: {linear_r2_scores}")
            st.write(f"Mean R² score: {linear_r2_scores.mean()}")

        if "XG Boost" in selected_models:
            st.header("XG Boost")
            code = """
            model_xg = xg.XGBRegressor(objective ='reg:linear',n_estimators = 2000, seed = 123)
            scores_xg = cross_val_score(model_xg, X_train, y_train, cv=10,scoring ='r2')
            """
            st.code(code, language='python')
            xg_r2_scores = np.load('scores/xg_r2_scores.npy')
            st.write(f"R² scores: {xg_r2_scores}")
            st.write(f"Mean R² score: {xg_r2_scores.mean()}")

        if "Lighboost" in selected_models:
            st.header("Lighboost")
            code = """
            params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'dart',
                    'num_leaves': 32,
                    'learning_rate': 0.3,
                    'feature_fraction': 0.9
                }
            bst = lgb.train(
                    params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                )
            y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
            lgb_r2 = r2_score(y_test, y_pred)
            """
            st.code(code, language='python')
            lgb_r2_scores = np.load('scores/bst_r2_scores.npy')
            st.write(f"R² scores: {lgb_r2_scores}")
            st.write(f"Mean R² score: {lgb_r2_scores.mean()}")

        if "Random Forest" in selected_models:
            st.header("Random Forest")
            code = """
            model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
            scores_rf = cross_val_score(model_rf, X_train, y_train, cv=10,scoring ='r2')
            """
            st.code(code, language='python')
            rf_r2_scores = np.load('scores/rf_r2_scores.npy')
            st.write(f"R² scores: {rf_r2_scores}")
            st.write(f"Mean R² score: {rf_r2_scores.mean()}")

        if "Catboost" in selected_models:
            st.header("Catboost")
            code = """
            model_cat = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE')
            scores_cat = cross_val_score(model_cat, X_train, y_train, cv=10,scoring ='r2')
            """
            st.code(code, language='python')
            cat_r2_scores = np.load('scores/cat_r2_scores.npy')
            st.write(f"R² scores: {cat_r2_scores}")
            st.write(f"Mean R² score: {cat_r2_scores.mean()}")

        if "Decision Tree Regression" in selected_models:
            st.header("Decision Tree Regression")
            code = """
            model_dt = DecisionTreeRegressor()
            scores_dt = cross_val_score(model_dt, X_train, y_train, cv=10,scoring ='r2')
            """
            st.code(code, language='python')
            dt_r2_scores = np.load('scores/dt_r2_scores.npy')
            st.write(f"R² scores: {dt_r2_scores}")
            st.write(f"Mean R² score: {dt_r2_scores.mean()}")

        if "Polynomial Regression" in selected_models:
            st.header("Polynomial Regression")
            code = """
            model_poly = LinearRegression()
            model_poly.fit(X_train, y_train)
            y_pred = model_poly.predict(X_test)
            poly_r2 = r2_score(y_test, y_pred)`
            """
            st.code(code, language='python')
            poly_r2_scores = np.load('scores/poly_r2_scores.npy')
            st.write(f"R² scores: {poly_r2_scores}")
            st.write(f"Mean R² score: {poly_r2_scores.mean()}")

        if "Support Vector Regressor" in selected_models:
            st.header("Support Vector Regressor")
            code = """
            model_svr = SVR(kernel='rbf', C=1000, epsilon=0.1)
            scores_svr = cross_val_score(model_svr, X_train, y_train, cv=5,scoring ='r2')
            """
            st.code(code, language='python')
            svr_r2_scores = np.load('scores/svr_r2_scores.npy')
            st.write(f"R² scores: {svr_r2_scores}")
            st.write(f"Mean R² score: {svr_r2_scores.mean()}")


    if st.session_state["Total Compare"]:
        st.header("Total Compare")

        scores_dict = {
            "Linear Regression": np.load('scores/linear_r2_scores.npy'),
            "XG Boost": np.load('scores/xg_r2_scores.npy'),
            "Lighboost": np.load('scores/bst_r2_scores.npy'),
            "Random Forest": np.load('scores/rf_r2_scores.npy'),
            "Catboost": np.load('scores/cat_r2_scores.npy'),
            "Decision Tree Regression": np.load('scores/dt_r2_scores.npy'),
            "Polynomial Regression": np.load('scores/poly_r2_scores.npy'),
            "Support Vector Regressor": np.load('scores/svr_r2_scores.npy')
        }

        selected_models = [model for model in model_options if st.session_state[model]]
        r2_scores = [scores_dict[model].mean() for model in selected_models]

        # Vẽ biểu đồ cột
        plt.figure(figsize=(10, 6))
        plt.bar(selected_models, r2_scores, color=['blue', 'green', 'red', 'purple', 'brown', 'orange', 'black', 'green'])

        # Thiết lập tiêu đề và nhãn
        plt.title('Comparison of R² Scores between Models')
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.xticks(rotation=30)

        # Hiển thị giá trị R² score trên đầu mỗi cột
        for i, score in enumerate(r2_scores):
            plt.text(i, score + 0.01, str(round(score, 3)), ha='center', va='bottom')

        # Hiển thị biểu đồ
        plt.ylim(0, 1.1)  # Đặt giới hạn trục y từ 0 đến 1
        st.pyplot(plt)

elif selected_main_menu == "Data Dashboard":
    st.header("Data Dashboard")

    # Select columns to visualize
    columns = raw.columns.tolist()
    x_axis = st.selectbox("Select X-axis:", columns)
    y_axis = st.selectbox("Select Y-axis:", columns)
    chart_type = st.selectbox("Select chart type:", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram"])
    group_by = st.selectbox("Select group by column (optional):", ["None"] + columns)

    if st.button("Generate Chart"):
        if group_by != "None":
            if chart_type == "Line Chart":
                fig = px.line(raw, x=x_axis, y=y_axis, color=group_by, title=f'{y_axis} over {x_axis} grouped by {group_by}')
            elif chart_type == "Bar Chart":
                fig = px.bar(raw, x=x_axis, y=y_axis, color=group_by, title=f'{y_axis} by {x_axis} grouped by {group_by}')
            elif chart_type == "Scatter Plot":
                fig = px.scatter(raw, x=x_axis, y=y_axis, color=group_by, title=f'{y_axis} vs {x_axis} grouped by {group_by}')
            elif chart_type == "Histogram":
                fig = px.histogram(raw, x=x_axis, color=group_by, title=f'{x_axis} Distribution grouped by {group_by}')
        else:
            if chart_type == "Line Chart":
                fig = px.line(raw, x=x_axis, y=y_axis, title=f'{y_axis} over {x_axis}')
            elif chart_type == "Bar Chart":
                fig = px.bar(raw, x=x_axis, y=y_axis, title=f'{y_axis} by {x_axis}')
            elif chart_type == "Scatter Plot":
                fig = px.scatter(raw, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
            elif chart_type == "Histogram":
                fig = px.histogram(raw, x=x_axis, title=f'{x_axis} Distribution')

        st.plotly_chart(fig)

