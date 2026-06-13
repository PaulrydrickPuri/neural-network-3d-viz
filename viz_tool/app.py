import streamlit as st
import numpy as np
from pathlib import Path
from viz_tool.model_loader import load_yolo_model
from viz_tool.gradcam import GradCAMExtractor
from viz_tool.feature_extractor import FeatureExtractor, project_to_3d
from viz_tool.visualization import (
    create_3d_feature_space,
    create_gradcam_overlay,
    create_training_metrics
)
from viz_tool.utils import load_image, get_image_paths, load_training_logs

st.set_page_config(page_title="NN Training Visualizer", layout="wide")
st.title("Neural Network 3D Training Visualization")

# Sidebar: Model Selection
st.sidebar.header("Model Configuration")
teacher_path = st.sidebar.text_input("Teacher Model Path (.pt)")
student_path = st.sidebar.text_input("Student Model Path (.pt)")
dataset_path = st.sidebar.text_input("Validation Dataset Path")

# Load models
teacher_model = load_yolo_model(teacher_path) if teacher_path else None
student_model = load_yolo_model(student_path) if student_path else None

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Architecture",
    "Grad-CAM Comparison",
    "Feature Space 3D",
    "Training Metrics"
])

# Tab 1: Architecture Viewer
with tab1:
    st.header("Model Architecture")
    if teacher_model:
        st.write("✅ Teacher model loaded successfully")
        st.write(f"Parameters: {sum(p.numel() for p in teacher_model.model.parameters()):,}")
    else:
        st.info("Load a teacher model to view architecture")

    if student_model:
        st.write("✅ Student model loaded successfully")
        st.write(f"Parameters: {sum(p.numel() for p in student_model.model.parameters()):,}")

# Tab 2: Grad-CAM Comparison
with tab2:
    st.header("Grad-CAM: Teacher vs Student")

    if teacher_model and student_model and dataset_path:
        # Get sample image
        images = get_image_paths(dataset_path, max_images=1)
        if images:
            img = load_image(images[0])
            class_idx = 0  # First class

            # Teacher Grad-CAM
            st.subheader("Computing Teacher Grad-CAM...")
            teacher_extractor = GradCAMExtractor(teacher_model)
            teacher_heatmap = teacher_extractor.compute_heatmap(img, class_idx)
            teacher_fig = create_gradcam_overlay(img, teacher_heatmap)

            # Student Grad-CAM
            st.subheader("Computing Student Grad-CAM...")
            student_extractor = GradCAMExtractor(student_model)
            student_heatmap = student_extractor.compute_heatmap(img, class_idx)
            student_fig = create_gradcam_overlay(img, student_heatmap)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Teacher")
                st.plotly_chart(teacher_fig, use_container_width=True)
            with col2:
                st.subheader("Student")
                st.plotly_chart(student_fig, use_container_width=True)
        else:
            st.warning("No images found in dataset path")
    else:
        st.info("Load both models and dataset to compare Grad-CAM")

# Tab 3: Feature Space 3D
with tab3:
    st.header("3D Feature Space")

    if teacher_model and student_model and dataset_path:
        images = get_image_paths(dataset_path, max_images=100)
        if images:
            st.info(f"Processing {len(images)} images...")

            # Extract features
            img_batch = [load_image(p) for p in images]

            st.subheader("Extracting Teacher Features...")
            teacher_extractor = FeatureExtractor(teacher_model)
            teacher_embeddings = teacher_extractor.extract(img_batch)
            teacher_3d = project_to_3d(teacher_embeddings)

            st.subheader("Extracting Student Features...")
            student_extractor = FeatureExtractor(student_model)
            student_embeddings = student_extractor.extract(img_batch)
            student_3d = project_to_3d(student_embeddings)

            # Create labels (simplified - use dummy labels)
            labels = np.zeros(len(images), dtype=int)
            class_names = ["Class 0"]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Teacher Feature Space")
                teacher_fig = create_3d_feature_space(teacher_3d, labels, class_names)
                st.plotly_chart(teacher_fig, use_container_width=True)
            with col2:
                st.subheader("Student Feature Space")
                student_fig = create_3d_feature_space(student_3d, labels, class_names)
                st.plotly_chart(student_fig, use_container_width=True)
        else:
            st.warning("No images found in dataset path")
    else:
        st.info("Load both models and dataset to visualize feature space")

# Tab 4: Training Metrics
with tab4:
    st.header("Training Metrics")
    log_path = st.text_input("Training Log Path (results.csv)")

    if log_path and Path(log_path).exists():
        logs = load_training_logs(log_path)
        metrics_fig = create_training_metrics(
            logs['epochs'],
            logs['loss'],
            logs['map50']
        )
        st.plotly_chart(metrics_fig, use_container_width=True)
    else:
        st.info("Provide path to training results.csv to view metrics")
