import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import torch
from ultralytics import YOLO
import shutil
import yaml
from streamlit_drawable_canvas import st_canvas

# Initialize models
@st.cache_resource
def load_models():
    models = {
        "YOLOv8n": YOLO('yolov8n.pt'),
        "YOLOv8s": YOLO('yolov8s.pt'),
        "YOLOv8m": YOLO('yolov8m.pt'),
    }
    return models

def process_image(image_path, model):
    try:
        image = cv2.imread(image_path)
        results = model(image)
        return results, image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def save_yolo_labels(labels, img_shape, output_path):
    try:
        img_height, img_width = img_shape[:2]
        with open(output_path, 'w') as f:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    except Exception as e:
        st.error(f"Error saving labels: {str(e)}")

def prepare_yolo_dataset(input_folder, output_folder, model):
    try:
        os.makedirs(output_folder, exist_ok=True)
        images_folder = os.path.join(output_folder, 'images')
        labels_folder = os.path.join(output_folder, 'labels')
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(labels_folder, exist_ok=True)

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return images_folder, labels_folder, image_files
    except Exception as e:
        st.error(f"Error preparing dataset: {str(e)}")
        return None, None, []

def create_data_yaml(output_folder, class_names):
    try:
        data_yaml = {
            'train': 'images',
            'val': 'images',
            'nc': len(class_names),
            'names': class_names
        }
        with open(os.path.join(output_folder, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f)
    except Exception as e:
        st.error(f"Error creating data.yaml: {str(e)}")

def initialize_session_state():
    if 'image_files' not in st.session_state:
        st.session_state['image_files'] = []
    if 'current_image_index' not in st.session_state:
        st.session_state['current_image_index'] = 0
    if 'input_folder' not in st.session_state:
        st.session_state['input_folder'] = ''
    if 'images_folder' not in st.session_state:
        st.session_state['images_folder'] = ''
    if 'labels_folder' not in st.session_state:
        st.session_state['labels_folder'] = ''
    if 'class_names' not in st.session_state:
        st.session_state['class_names'] = []
    if 'canvas_key' not in st.session_state:
        st.session_state['canvas_key'] = 0

def main():
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

            # Manual labeling
            st.write("Add custom label:")
            custom_class = st.text_input("Class name:")

            if st.button("Add Custom Label"):
                if custom_class:
                    if custom_class not in st.session_state['class_names']:
                        st.session_state['class_names'].append(custom_class)
                    class_id = st.session_state['class_names'].index(custom_class)
                    # Append manual bounding box with the custom class
                    detected_objects.append((class_id, x_center, y_center, width, height))
                    st.success(f"Added custom label: {custom_class}")

            if st.button("Save and Next"):
                # Save labels
                output_label_path = os.path.join(st.session_state['labels_folder'], os.path.splitext(current_image)[0] + '.txt')
                save_yolo_labels(detected_objects, image.size, output_label_path)

                # Copy image
                try:
                    shutil.copy(image_path, os.path.join(st.session_state['images_folder'], current_image))
                except Exception as e:
                    st.error(f"Error copying image: {str(e)}")

                # Move to next image
                st.session_state['current_image_index'] += 1
                st.session_state['canvas_key'] += 1

                # Update progress bar
                progress.progress(st.session_state['current_image_index'] / len(st.session_state['image_files']))

                if st.session_state['current_image_index'] >= len(st.session_state['image_files']):
                    st.success("All images processed!")
                    create_data_yaml(output_folder, st.session_state['class_names'])
                    st.write("Dataset preparation complete. You can now proceed with YOLO training.")
                
                st.rerun()
        else:
            st.success("All images have been processed. You can now proceed with YOLO training.")

    st.write("---")
    st.write("YOLO Training")

    # Training parameters
    epochs = st.number_input("Number of epochs", min_value=1, value=100)
    batch_size = st.number_input("Batch size", min_value=1, value=16)
    imgsz = st.number_input("Image size", min_value=32, value=640)

    if st.button("Start Training"):
        if output_folder:
            with st.spinner("Training YOLO model..."):
                try:
                    # Initialize a new YOLO model for training
                    model = YOLO('yolov8n.yaml')  # Create a new model from scratch
                    
                    # Train the model
                    results = model.train(
                        data=os.path.join(output_folder, 'data.yaml'),
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        name='yolo_custom_model'
                    )
                    
                    st.success("Training complete! Model weights saved in 'runs/detect/yolo_custom_model' directory.")
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
        else:
            st.error("Please process the dataset first.")

    if st.button("Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Application state has been reset. You can start over with a new dataset.")
        st.rerun()

if __name__ == "__main__":
    main()