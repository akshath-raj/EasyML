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
from rm import run_all_classifiers ,preprocess_data_with_sampling
import matplotlib.pyplot as plt
import seaborn as sns
from hp import custom_RandomForestClassifier,evaluate_random_forest
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas
import shutil
from yolo_utils import load_models, process_image, prepare_yolo_dataset, save_yolo_labels, create_data_yaml, initialize_session_state

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
add_bg_from_local('Untitled design (6).jpg')  # Ensure correct file path

# Function to upload the file
def upload_file():
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df  # Save the dataframe in session state
        st.success("File uploaded successfully!")
    else:
        st.warning("Please upload a CSV file.")

# Function to display data without the column selection
def display_data():
    if 'df' in st.session_state:
        df = st.session_state['df']  # Retrieve the dataframe from session state
        st.write(df)  # Display the dataframe
    else:
        st.warning("No file uploaded yet. Please upload a file.")

# Creating a horizontal menu bar
selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Data", "Analysis", "Eda","Yolo"],  # required
    icons=["house", "bar-chart", "clipboard-data", "info","search"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",  # can also be "vertical"
)

# Based on the selected option, display different pages
if selected == "Home":
    st.title("Welcome to Error 404-EasyML")
    st.write("This is the home page of the app.")
    upload_file()
def add_column_with_formula(df, new_column_name, formula):
    try:
        # Evaluate the formula in the context of the DataFrame
        df[new_column_name] = df.eval(formula)
        print(f"Column '{new_column_name}' added successfully!")
    except Exception as e:
        print(f"Error: {e}")
    return df
if selected == "Data":
    st.title("Data Page")
    
    # Center the dataset table using CSS
    st.markdown("""
    <style>
    .center-table {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the dataset in the center
    with st.container():
        st.write('<div class="center-table">', unsafe_allow_html=True)
        
        # Display the existing DataFrame from session state
        if 'df' in st.session_state:
            st.dataframe(st.session_state['df'])
        st.write('</div>', unsafe_allow_html=True)
    
    # Create two columns for the dropdown and radio button
    col1, col2 = st.columns(2)
    
    # Column 1: Dropdown for selecting a column to predict
    with col1:
        st.subheader("Select Column to Predict")
        if 'df' in st.session_state:
            df = st.session_state['df']
            column = st.selectbox("Choose the target column:", df.columns)
            st.session_state['target_column'] = column

    # Column 2: Radio button for selecting operation (regression/classification)
    with col2:
        st.subheader("Select Your Operation")
        genre = st.radio(
            "Choose an operation:",
            ["Regression", "Classification"]
        )
    
        # Store the selected task (regression/classification) in session state
        if genre == "Regression":
            st.session_state['task'] = 'regression'
        elif genre == "Classification":
            st.session_state['task'] = 'classification'
    
    # Section to add a new column with a user-defined formula
    st.subheader("Add a New Column with a Formula")
    vvc, vvcc = st.columns(2)
    new_column_name = vvc.text_input("Enter the new column name:")
    formula = vvcc.text_input("Enter the formula (use column names):")

    # Button to apply the formula
    if st.button("Add Column"):
        if new_column_name and formula:
            try:
                # Apply the formula and add the new column
                st.session_state['df'] = add_column_with_formula(st.session_state['df'], new_column_name, formula)
                st.success(f"Column '{new_column_name}' added successfully!")
                
                # Display the updated DataFrame
                st.dataframe(st.session_state['df'])  # Refresh the DataFrame display
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide both a column name and a formula.")
    
    st.subheader("Hyperparameter Tuning")
    vvc1, vcc2, vcc3, vcc4 = st.columns(4)
    
    # Input fields for hyperparameters
    n_estimators1 = vvc1.text_input("n_estimators", placeholder="100")
    max_depth1 = vcc2.text_input("max_depth", placeholder="None")
    max_features1 = vcc3.text_input("max_features", placeholder="sqrt")
    verbose1 = vcc4.text_input("verbose", placeholder="0")
    
    if st.button("Enable Hyperparameters"):
        try:
            # Validate inputs before passing to the classifier
            n_estimators = int(n_estimators1) if n_estimators1 else 100
            max_depth = None if max_depth1 == "" else int(max_depth1)  # Allow for None
            max_features = max_features1 if max_features1 in {'sqrt', 'log2', None} or max_features1.isdigit() else 'sqrt'
            verbose = int(verbose1) if verbose1 else 0

            # Create the RandomForestClassifier
            mom = custom_RandomForestClassifier(
                n_estimators=n_estimators,
                criterion='gini',
                max_depth=max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=max_features,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=None,
                random_state=None,
                verbose=verbose,
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None
            )

            xt1, xt2, xt3, xt4 = preprocess_data_with_sampling(st.session_state['df'], st.session_state['target_column'])
            r23 = evaluate_random_forest(xt1, xt2, xt3, xt4, mom)
            st.session_state['r23'] = r23
            st.write("r23 updated")
        except ValueError as ve:
            st.error(f"Value error: {ve}")
        except Exception as e:
            st.error(f"Error: {e}")

elif selected == "Analysis":
    st.title("Analysis Page")
    st.write("Perform data analysis here.")

    if 'df' in st.session_state and 'target_column' in st.session_state:
        df = st.session_state['df']
        target_column = st.session_state['target_column']
        
        if 'task' in st.session_state:
            if st.session_state['task'] == 'regression':
                ty = run_all_regressors(df, target_column)
                new = pd.DataFrame.from_dict(ty)
                st.write(new.transpose())
            elif st.session_state['task'] == 'classification':
                ty = run_all_classifiers(df, target_column)
                new = pd.DataFrame.from_dict(ty)
                st.write(new.transpose())

        # Display r23 if it exists
        if 'r23' in st.session_state:
            st.write("Current r23 value:", st.session_state['r23'])  # Debugging output

            
            #new2 = pd.DataFrame(st.session_state['r23'])
        
        
            
            
            #st.write(new2.transpose())
        else:
            st.warning("Please run the model on the Data page to see results.")
    else:
        st.warning("Please upload a CSV file and select a target column on the Data page.")

elif selected == "Eda":
    st.title("EDA - Exploratory Data Analysis")

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

        # Function to get and plot correlation matrix for numeric columns (adjusted to smaller size)
        def correlation_matrix(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return None
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(5, 4))  # Adjust size to make it smaller
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            plt.title('Correlation Matrix')
            return fig

        # Function to plot histograms for numerical features
        def plot_histograms(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                fig, axes = plt.subplots(nrows=(len(numeric_cols) + 1) // 2, ncols=2, figsize=(14, 4 * ((len(numeric_cols) + 1) // 2)))
                axes = axes.flatten()

                for i, col in enumerate(numeric_cols):
                    df[col].plot(kind='hist', bins=30, ax=axes[i], title=f'Histogram of {col}', color='blue', edgecolor='black')
                    axes[i].set_xlabel(col)

                for ax in axes[len(numeric_cols):]:
                    fig.delaxes(ax)

                plt.tight_layout()
                return fig
            else:
                st.warning("No numeric columns found for plotting histograms.")
                return None

        # Function to plot KDE for numerical features
        def plot_kde(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.kdeplot(df[col], ax=ax, fill=True, color='blue')
                    ax.set_title(f'KDE Plot of {col}')
                    ax.set_xlabel(col)
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting KDE.")
                return None

        # Function to plot boxplots for numerical features
        def plot_boxplots(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f'Boxplot of {col}')
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting boxplots.")
                return None

        # Function to plot violin plots for numerical features
        def plot_violinplots(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                figs = []
                for col in numeric_cols:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.violinplot(x=df[col], ax=ax)
                    ax.set_title(f'Violin Plot of {col}')
                    figs.append(fig)
                return figs
            else:
                st.warning("No numeric columns found for plotting violin plots.")
                return None

        # Function to plot pair plots for numeric features
        def plot_pairplot(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                fig = sns.pairplot(df[numeric_cols])
                plt.suptitle("Pair Plot", y=1.02)
                return fig
            else:
                st.warning("Not enough numeric columns to create pair plot.")
                return None

        # Function to plot count plots for categorical features
        def plot_countplots(df):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                figs = []
                for col in categorical_cols:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
                    plt.title(f'Countplot of {col}')
                    plt.xticks(rotation=90)
                    figs.append(fig)
                return figs
            else:
                st.warning("No categorical columns found for count plots.")
                return None

        # Layout the page with a sidebar for options and the graphs in a 2-column layout
        with st.sidebar:
            st.header("EDA Options")
            sel = option_menu(
                menu_title=None,  # required
                options=["Correlation Matrix", "Histograms", "KDE Plots", "Boxplots", "Violin Plots", "Pair Plot", "Count Plots","Dataset info"],  # required
                icons=["graph", "graph", "graph", "graph", "graph", "graph", "graph","graph"],
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="vertical"  # Sidebar will be vertical
            )

        # Use two columns for displaying graphs (side by side)
        col1, col2 = st.columns(2)

        # Display correlation matrix
        if sel == "Correlation Matrix":
            corr_fig = correlation_matrix(df)
            if corr_fig:
                with st.container():
                    st.write("### Correlation Matrix:")
                    st.pyplot(corr_fig)

        # Display histograms
        if sel == "Histograms":
            hist_fig = plot_histograms(df)
            if hist_fig:
                with st.container():
                    st.write("### Histograms of Numeric Features:")
                    st.pyplot(hist_fig)

        # Display KDE plots
        if sel == "KDE Plots":
            kde_plots = plot_kde(df)
            if kde_plots:
                with st.container():
                    st.write("### KDE Plots of Numeric Features:")
                    for i in range(0, len(kde_plots), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(kde_plots[i])
                        if i + 1 < len(kde_plots):
                            with col2:
                                st.pyplot(kde_plots[i + 1])

        # Display boxplots
        if sel == "Boxplots":
            boxplots = plot_boxplots(df)
            if boxplots:
                with st.container():
                    st.write("### Boxplots of Numeric Features:")
                    for i in range(0, len(boxplots), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(boxplots[i])
                        if i + 1 < len(boxplots):
                            with col2:
                                st.pyplot(boxplots[i + 1])

        # Display violin plots
        if sel == "Violin Plots":
            violin_plots = plot_violinplots(df)
            if violin_plots:
                with st.container():
                    st.write("### Violin Plots of Numeric Features:")
                    for i in range(0, len(violin_plots), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(violin_plots[i])
                        if i + 1 < len(violin_plots):
                            with col2:
                                st.pyplot(violin_plots[i + 1])

        # Display pair plot
        if sel == "Pair Plot":
            pair_plot = plot_pairplot(df)
            if pair_plot:
                with st.container():
                    st.write("### Pair Plot of Numeric Features:")
                    st.pyplot(pair_plot)

        # Display count plots
        if sel == "Count Plots":
            count_plots = plot_countplots(df)
            if count_plots:
                with st.container():
                    st.write("### Count Plots of Categorical Features:")
                    for i in range(0, len(count_plots), 2):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(count_plots[i])
                        if i + 1 < len(count_plots):
                            with col2:
                                st.pyplot(count_plots[i + 1])


        if sel=="Dataset info":
          st.write("### Dataset Information")
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
elif selected=="Yolo":
      st.title("YOLO Dataset Preparation with Manual Labeling")

      initialize_session_state()

      models = load_models()

      # Model selection
      selected_model = st.selectbox("Choose a model for initial labeling", list(models.keys()))

      # Folder selection
      input_folder = st.text_input("Enter the path to your input image folder:")
      output_folder = st.text_input("Enter the path for the output dataset:")

      if st.button("Prepare Dataset"):
          if input_folder and output_folder:
              with st.spinner("Preparing dataset structure..."):
                  images_folder, labels_folder, image_files = prepare_yolo_dataset(input_folder, output_folder, models[selected_model])
              if images_folder and labels_folder:
                  st.session_state['image_files'] = image_files
                  st.session_state['input_folder'] = input_folder
                  st.session_state['images_folder'] = images_folder
                  st.session_state['labels_folder'] = labels_folder
                  st.session_state['current_image_index'] = 0
                  st.session_state['class_names'] = list(models[selected_model].names.values())
                  st.success("Dataset structure prepared. Ready for labeling.")
              else:
                  st.error("Failed to prepare dataset structure.")

      if len(st.session_state['image_files']) > 0:
          st.write("---")
          st.write("Manual Labeling")

          # Progress bar
          progress = st.progress(0)

          if st.session_state['current_image_index'] < len(st.session_state['image_files']):
              current_image = st.session_state['image_files'][st.session_state['current_image_index']]
              st.write(f"Current Image: {current_image} ({st.session_state['current_image_index'] + 1}/{len(st.session_state['image_files'])})")

              # Display the image
              image_path = os.path.join(st.session_state['input_folder'], current_image)
              try:
                  image = Image.open(image_path)
                  st.image(image, use_column_width=True)
              except Exception as e:
                  st.error(f"Error opening image: {str(e)}")
                  st.stop()

              # Perform initial detection
              results, _ = process_image(image_path, models[selected_model])

              # Display detected objects
              detected_objects = []
              if results:
                  for r in results:
                      boxes = r.boxes.xyxy.cpu().numpy()
                      classes = r.boxes.cls.cpu().numpy()
                      for box, cls in zip(boxes, classes):
                          x1, y1, x2, y2 = box
                          class_name = models[selected_model].names[int(cls)]
                          detected_objects.append((int(cls), (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))
                          st.write(f"Detected: {class_name} at {box}")

              # Canvas for drawing manual bounding boxes
              canvas_result = st_canvas(
                  fill_color="rgba(255, 165, 0, 0.3)",
                  stroke_width=2,
                  background_image=image,
                  update_streamlit=True,
                  height=image.height,
                  width=image.width,
                  drawing_mode="rect",
                  key=f"canvas_{st.session_state['canvas_key']}",
              )

              # Process manually drawn bounding box
              if canvas_result.json_data is not None:
                  objects = canvas_result.json_data["objects"]
                  for obj in objects:
                      x_center = (obj['left'] + obj['width'] / 2) / image.width
                      y_center = (obj['top'] + obj['height'] / 2) / image.height
                      width = obj['width'] / image.width
                      height = obj['height'] / image.height
                      class_id = 0 
                      detected_objects.append((class_id, x_center, y_center, width, height))
                      st.write(f"Manually drawn bounding box: x_center={x_center}, y_center={y_center}, width={width}, height={height}")

              if st.button("Save and Next"):
                  output_label_path = os.path.join(st.session_state['labels_folder'], os.path.splitext(current_image)[0] + '.txt')
                  save_yolo_labels(detected_objects, image.size, output_label_path)

                  try:
                      shutil.copy(image_path, os.path.join(st.session_state['images_folder'], current_image))
                  except Exception as e:
                      st.error(f"Error copying image: {str(e)}")

                  st.session_state['current_image_index'] += 1
                  st.session_state['canvas_key'] += 1
                  progress.progress(st.session_state['current_image_index'] / len(st.session_state['image_files']))

                  if st.session_state['current_image_index'] >= len(st.session_state['image_files']):
                      st.success("All images processed!")
                      create_data_yaml(output_folder, st.session_state['class_names'])
                      st.write("Dataset preparation complete. You can now proceed with YOLO training.")
                  
                  st.rerun()
          else:
              st.success("All images have been processed.")

      st.write("---")
      st.write("YOLO Training")

      epochs = st.number_input("Number of epochs", min_value=1, value=100)
      batch_size = st.number_input("Batch size", min_value=1, value=16)
      imgsz = st.number_input("Image size", min_value=32, value=640)

      if st.button("Start Training"):
          if output_folder:
              with st.spinner("Training YOLO model..."):
                  try:
                      model = models[selected_model]
                      results = model.train(
                          data=os.path.join(output_folder, 'data.yaml'),
                          epochs=epochs,
                          imgsz=imgsz,
                          batch=batch_size,
                          name='yolo_custom_model'
                      )
                      st.success("Training complete! Model weights saved.")
                  except Exception as e:
                      st.error(f"Error during training: {str(e)}")
          else:
              st.error("Please process the dataset first.")

      if st.button("Reset"):
          for key in list(st.session_state.keys()):
              del st.session_state[key]
          st.success("Application state reset.")
          st.rerun()
