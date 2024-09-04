import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from io import StringIO

# Set the OpenAI API key
# choo nay tu them APIkey

# Streamlit App configuration
st.set_page_config(
    page_title="Carbon Majors Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to initialize session state
def initialize_state(state_name, default_value=False):
    if state_name not in st.session_state:
        st.session_state[state_name] = default_value

# Initialize session state for all options
options = [
    "show_general_info",
    "show_missing_values",
    "show_statistics",
    "show_emissions_year",
    "show_emissions_parent_type",
    "show_emissions_commodity",
    "show_production_vs_emissions",
    "show_correlation_heatmap",
    "llm_response"
]

for option in options:
    initialize_state(option)

# App title and description
st.title("Carbon Majors Data Analysis")
st.markdown("""
    Welcome to the Carbon Majors Data Analysis app. Upload your CSV or Excel file and explore various aspects of the dataset
    including data summary, missing values, statistical analysis, and visualizations of emissions data.
""")

# File Upload
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
def design_prompt(variable):
    prompt = f"Analyzing data with the following input: {variable}"
    return prompt

if uploaded_file is not None:
    try:
        # Read uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        # Display top 10 rows of the DataFrame
        st.subheader("Top 10 Rows of the Dataset")
        st.dataframe(df.head(10), use_container_width=True)

        # User Input for LLM
        st.subheader("Interact with Your Data")
        st.markdown("Enter your prompt to interact with the data using OpenAI.")
        variable = st.text_area("Enter your prompt:")

        if st.button("Chat with CSV"):
            if variable:
            # try:
            #     llm = OpenAI()
            #     pandas_ai = SmartDataframe(df, config={"llm": llm})
            #     result = pandas_ai.chat(variable)
            #     st.session_state.llm_response = result
            #     st.success(result)
            # except Exception as e:
            #     st.error(f"An error occurred while interacting with the data: {e}")
                try:
                    # Thiết kế prompt từ đầu vào của người dùng
                    prompt = design_prompt(variable)
                    
                    # Khởi tạo đối tượng OpenAI
                    llm =  # Thay thế YOUR_OPENAI_API_KEY bằng API key thực tế
                    
                    # Khởi tạo SmartDataframe với cấu hình LLM
                    pandas_ai = SmartDataframe(df, config={"llm": llm})
                    
                    # Gọi hàm chat với prompt đã thiết kế
                    result = pandas_ai.chat(prompt)

                    if isinstance(result, pd.DataFrame):
                        result_str = result.to_string(index=True)
                    else:
                        result_str = str(result)
                    
                    # Lưu kết quả vào session state
                    st.session_state.llm_response = result_str
                
                except Exception as e:
                    st.error(f"An error occurred while interacting with the data: {e}")
        if st.button("Clear Result"):
            st.session_state.llm_response = ""
            st.session_state.prompt_input = ""

        if st.session_state.llm_response:
            st.subheader("Result")
            st.text(st.session_state.llm_response)
        
        # elif st.session_state.llm_response:
        #     st.text(st.session_state.llm_response)
        
        # Sidebar for data analysis options
        st.sidebar.header("Data Analysis Options")

        st.sidebar.checkbox("General Info", key="show_general_info")
        st.sidebar.checkbox("Missing Values", key="show_missing_values")
        st.sidebar.checkbox("Basic Statistics", key="show_statistics")
        st.sidebar.checkbox("Emissions by Year", key="show_emissions_year")
        st.sidebar.checkbox("Emissions by Company Type", key="show_emissions_parent_type")
        st.sidebar.checkbox("Emissions by Commodity", key="show_emissions_commodity")
        st.sidebar.checkbox("Production vs. Emissions", key="show_production_vs_emissions")
        st.sidebar.checkbox("Correlation Heatmap", key="show_correlation_heatmap")

        # Display analysis based on user selection
        if st.session_state.show_general_info:
            st.subheader("General Information")
            try:
                buffer = StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.write(df.describe())
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_missing_values:
            st.subheader("Missing Values")
            try:
                st.write(df.isnull().sum())
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_statistics:
            st.subheader("Basic Statistics")
            try:
                st.write(df.describe())
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_emissions_year:
            st.subheader("Emissions by Year")
            try:
                total_emissions_per_year = df.groupby('year')['total_emissions_MtCO2e'].sum().reset_index()
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=total_emissions_per_year, x='year', y='total_emissions_MtCO2e')
                plt.title('Total Emissions by Year')
                plt.xlabel('Year')
                plt.ylabel('Total Emissions (MtCO2e)')
                st.pyplot(plt)
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_emissions_parent_type:
            st.subheader("Emissions by Company Type")
            try:
                total_emissions_per_parent_type = df.groupby('parent_type')['total_emissions_MtCO2e'].sum().reset_index()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=total_emissions_per_parent_type, x='parent_type', y='total_emissions_MtCO2e')
                plt.title('Total Emissions by Company Type')
                plt.xlabel('Company Type')
                plt.ylabel('Total Emissions (MtCO2e)')
                st.pyplot(plt)
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_emissions_commodity:
            st.subheader("Emissions by Commodity")
            try:
                total_emissions_per_commodity = df.groupby('commodity')['total_emissions_MtCO2e'].sum().reset_index()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=total_emissions_per_commodity, x='commodity', y='total_emissions_MtCO2e')
                plt.title('Total Emissions by Commodity')
                plt.xlabel('Commodity')
                plt.ylabel('Total Emissions (MtCO2e)')
                plt.xticks(rotation=45)
                st.pyplot(plt)
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_production_vs_emissions:
            st.subheader("Production vs. Emissions")
            try:
                plt.figure(figsize=(12, 6))
                sns.scatterplot(data=df, x='production_value', y='total_emissions_MtCO2e', hue='commodity')
                plt.title('Production vs. Emissions')
                plt.xlabel('Production Value')
                plt.ylabel('Total Emissions (MtCO2e)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(plt)
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")

        if st.session_state.show_correlation_heatmap:
            st.subheader("Correlation Heatmap")
            try:
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                corr = df[numeric_cols].corr()
                plt.figure(figsize=(14, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title('Correlation Heatmap')
                st.pyplot(plt)
            except Exception as e:
                st.error("Tính năng này không phù hợp với dữ liệu của bạn, bạn có thể sử dụng prompt để thực hiện yêu cầu tương tự đối với bộ dữ liệu của bạn.")
                
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")