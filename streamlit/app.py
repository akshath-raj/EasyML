import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import base64
import shap
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from be import run_all_regressors
from rm import run_all_classifiers 
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page layout to wide
st.set_page_config(layout="wide")

# Set dark mode
st.markdown(
    """
    <style>
        .stApp {
            background-color: #2E2E2E;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to add background image from a local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{img_base64});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Style for the app title */
        .stTitle {{
            font-family: 'Courier New', monospace;
            font-size: 2.5em;
            font-weight: bold;
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }}

        /* Customize menu bar */
        .nav {{
            background-color: rgba(255, 255, 255, 0.2);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }}

        .nav a {{
            color: white;
            font-size: 1.2em;
            padding: 10px;
            text-decoration: none;
            transition: 0.3s;
        }}

        .nav a:hover {{
            color: #f9c74f;
        }}

        /* Styling buttons */
        .stButton button {{
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1.2em;
        }}

        .stButton button:hover {{
            background-color: #0056b3;
            transition: 0.3s;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background image
add_bg_from_local('Untitled design (6).png')  # Ensure correct file path

# Function to upload the file
def upload_file():
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Save the dataframe in session state
        st.success("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file.")

# Function to display data
def display_data():
    if 'df' in st.session_state:
        df = st.session_state['df']  # Retrieve the dataframe from session state
        st.write(df)  # Display the dataframe
        
        # Dropdown for selecting a column
        column = st.selectbox("Select a column to predict", df.columns)
        
        # Store the selected column in session state
        st.session_state['target_column'] = column
        
        # Display selected column data
        st.write(f"Selected Column: {column}")
    else:
        st.warning("No file uploaded yet. Please upload a file.")

# Creating a horizontal menu bar
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Data", "Analysis", "Eda"],  # required
    icons=["house", "bar-chart", "clipboard-data", "info"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",  # can also be "vertical"
)

# Based on the selected option, display different pages
if selected == "Home":
    st.title("Welcome to the Home Page")
    st.write("This is the home page of the app.")
    upload_file()

elif selected == "Data":
    st.title("Data Page")
    st.write("Your DataSet")
    display_data()
    
    genre = st.radio(
        "Select your operation",
        ["Regression", "Classification"]
    )
    
    # Store the selected task (regression/classification) in session state
    if genre == "Regression":
        st.session_state['task'] = 'regression'
    elif genre == "Classification":
        st.session_state['task'] = 'classification'

elif selected == "Analysis":
    st.title("Analysis Page")
    st.write("Perform data analysis here.")
    
    # Only run analysis if a file and target column are selected
    if 'df' in st.session_state and 'target_column' in st.session_state:
        df = st.session_state['df']
        target_column = st.session_state['target_column']
        
        # Check task (regression or classification) selected by the user
        if 'task' in st.session_state:
            if st.session_state['task'] == 'regression':
                ty = run_all_regressors(df, target_column)
                new = pd.DataFrame.from_dict(ty)
                st.write(new.transpose())
            elif st.session_state['task'] == 'classification':
                ty = run_all_classifiers(df, target_column)
                new = pd.DataFrame.from_dict(ty)
                st.write(new.transpose())
        else:
            st.warning("Please select an operation on the Data page.")
    else:
        st.warning("Please upload a CSV file and select a target column on the Data page.")

elif selected == "Eda":
    st.title("Eda")
    
    if 'df' in st.session_state:  # Ensure df is available
        df = st.session_state['df']  # Retrieve the dataframe from session state

        # Function to get basic dataset information
        def dataset_info(df):
            info = {}
            info["shape"] = df.shape
            info["data_types"] = df.dtypes
            info["first_5_rows"] = df.head()
            info["missing_values"] = df.isnull().sum()
            info["unique_values"] = {col: df[col].nunique() for col in df.columns}
            return info

        # Function to get summary statistics
        def summary_statistics(df):
            return df.describe()

        # Function to get and plot correlation matrix for numeric columns
        def correlation_matrix(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return None
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            plt.title('Correlation Matrix')
            return fig

        # Function to plot histograms for numerical features
        def plot_histograms(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
            if numeric_cols:  # Only proceed if numeric columns are present
                num_plots = len(numeric_cols)
                fig, axes = plt.subplots(nrows=(num_plots + 1) // 2, ncols=2, figsize=(14, 4 * ((num_plots + 1) // 2)))  # Adjust subplots
                
                axes = axes.flatten()  # Flatten the axes array for easier iteration
                
                for i, col in enumerate(numeric_cols):
                    df[col].plot(kind='hist', bins=30, ax=axes[i], title=f'Histogram of {col}', color='blue', edgecolor='black')
                    axes[i].set_xlabel(col)
                
                # Remove any unused axes
                for ax in axes[num_plots:]:
                    fig.delaxes(ax)
                
                plt.tight_layout()
                return fig
            else:
                st.warning("No numeric columns found for plotting histograms.")
                return None

        # Function to plot boxplots for numerical features
        def plot_boxplots(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f'Boxplot of {col}')
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting boxplots.")
                return None

        # Function to plot KDE for numerical features
        def plot_kde(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.kdeplot(df[col], ax=ax, fill=True, color='blue')
                    ax.set_title(f'KDE Plot of {col}')
                    ax.set_xlabel(col)
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting KDE.")
                return None

        # Function to plot pair plots for numeric features
        def plot_pairplot(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
            if len(numeric_cols) > 1:  # Ensure there are at least 2 numeric columns for pair plot
                fig = sns.pairplot(df[numeric_cols])
                plt.suptitle("Pair Plot", y=1.02)
                return fig
            else:
                st.warning("Not enough numeric columns to create pair plot.")
                return None

        # Function to plot count plots for categorical features
        def plot_countplots(df):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()  # Get categorical columns
            if categorical_cols:
                figs = []
                for col in categorical_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
                    plt.title(f'Countplot of {col}')
                    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
                    figs.append(fig)
                return figs
            else:
                st.warning("No categorical columns found for count plots.")
                return None

        # Function to plot violin plots for numeric features
        def plot_violinplots(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric columns
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.violinplot(x=df[col], ax=ax)
                    ax.set_title(f'Violin Plot of {col}')
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting violin plots.")
                return None

        st.header("Dataset Information")
        info = dataset_info(df)
        st.write("Shape:", info["shape"])
        st.write("Data Types:")
        st.write(info["data_types"])
        st.write("First 5 rows:")
        st.write(info["first_5_rows"])
        st.write("Missing Values:")
        st.write(info["missing_values"])
        st.write("Unique Values:")
        st.write(info["unique_values"])

        # Summary statistics
        st.header("Summary Statistics")
        st.write(summary_statistics(df))


        # Display correlation matrix
        corr_fig = correlation_matrix(df)
        if corr_fig:
            st.write("### Correlation Matrix:")
            st.pyplot(corr_fig)

        # Display histograms
        hist_fig = plot_histograms(df)
        if hist_fig:
            st.write("### Histograms of Numeric Features:")
            st.pyplot(hist_fig)

        # Display boxplots
        boxplots = plot_boxplots(df)
        if boxplots:
            st.write("### Boxplots of Numeric Features:")
            cols = st.columns(len(boxplots))  # Create columns for boxplots
            for i, fig in enumerate(boxplots):
                with cols[i]:
                    st.pyplot(fig)

        # Display KDE plots
        kde_plots = plot_kde(df)
        if kde_plots:
            st.write("### KDE Plots of Numeric Features:")
            cols = st.columns(len(kde_plots))  # Create columns for KDE plots
            for i, fig in enumerate(kde_plots):
                with cols[i]:
                    st.pyplot(fig)

        # Display Pair Plot
        pair_plot = plot_pairplot(df)
        if pair_plot:
            st.write("### Pair Plot of Numeric Features:")
            st.pyplot(pair_plot)

        # Display count plots
        count_plots = plot_countplots(df)
        if count_plots:
            st.write("### Count Plots of Categorical Features:")
            cols = st.columns(len(count_plots))  # Create columns for count plots
            for i, fig in enumerate(count_plots):
                with cols[i]:
                    st.pyplot(fig)

        # Display violin plots
        violin_plots = plot_violinplots(df)
        if violin_plots:
            st.write("### Violin Plots of Numeric Features:")
            cols = st.columns(len(violin_plots))  # Create columns for violin plots
            for i, fig in enumerate(violin_plots):
                with cols[i]:
                    st.pyplot(fig)

        