# Plan: Neural Network 3D Training Visualization (Streamlit MVP)

**Goal:** Build a Python-only Streamlit dashboard that visualizes YOLO26 model training with 3D feature space, Grad-CAM comparisons, and real-time metrics.

**Tech stack:** Python 3.10+ / Streamlit / Plotly / PyTorch / Ultralytics YOLO

**Related skills needed:** tdd (for each component)

**Estimated tasks:** 8

---

## Assumptions

- User has a trained YOLO26 teacher model (.pt file) and student model (.pt file)
- User has a validation dataset (images + labels) accessible locally
- User is running on a machine with GPU (for Grad-CAM computation)
- Streamlit app will run on localhost:8501
- Models are PyTorch-based (Ultralytics YOLO format)

If any assumption is wrong, the plan needs adjustment for:
- Different model format (ONNX, TensorFlow)
- Remote dataset (S3, cloud storage)
- CPU-only machine (slower Grad-CAM)
- Different port or deployment target

---

## Simpler Alternative Considered

**Jupyter Notebook with Plotly:** Could build all visualizations in a single notebook without Streamlit UI. Rejected because:
- User wants interactive dashboard, not static notebook
- Streamlit provides better UX for sharing with teammates
- Easier to extend to React Three Fiber later

**Streamlit MVP is the simplest approach that meets requirements.**

---

## File Map

```
CREATE  viz_tool/
CREATE  viz_tool/__init__.py
CREATE  viz_tool/app.py                          # Main Streamlit app
CREATE  viz_tool/model_loader.py                 # Load YOLO models
CREATE  viz_tool/gradcam.py                      # Grad-CAM computation
CREATE  viz_tool/feature_extractor.py            # Extract embeddings
CREATE  viz_tool/visualization.py                # Plotly charts
CREATE  viz_tool/utils.py                        # Helper functions
CREATE  viz_tool/tests/
CREATE  viz_tool/tests/__init__.py
CREATE  viz_tool/tests/test_model_loader.py
CREATE  viz_tool/tests/test_gradcam.py
CREATE  viz_tool/tests/test_feature_extractor.py
CREATE  viz_tool/tests/test_visualization.py
CREATE  requirements-viz.txt                     # Dependencies
CREATE  docs/plans/neural-network-3d-viz.md      # This file
CREATE  docs/checklists/neural-network-3d-viz-checklist.md
```

---

## Task Breakdown

### Task 1: Project Setup & Dependencies
**Files:** `requirements-viz.txt`, `viz_tool/__init__.py`

**Success criteria:** `pip install -r requirements-viz.txt` succeeds without errors

**Steps:**
1. Create `requirements-viz.txt` with pinned versions:
   ```
   streamlit==1.31.0
   plotly==5.18.0
   torch>=2.0.0
   torchvision>=0.15.0
   ultralytics>=8.1.0
   opencv-python==4.9.0.81
   numpy==1.26.3
   pillow==10.2.0
   scikit-learn==1.4.0
   umap-learn==0.5.5
   ```

2. Create `viz_tool/__init__.py`:
   ```python
   """Neural Network 3D Training Visualization Tool."""
   __version__ = "0.1.0"
   ```

3. Install dependencies: `pip install -r requirements-viz.txt`

4. Verify installation: `python -c "import streamlit, plotly, torch, ultralytics; print('OK')"`

5. `git commit -m "feat: add project setup and dependencies for viz tool"`

---

### Task 2: Model Loader Module
**Files:** `viz_tool/model_loader.py`, `viz_tool/tests/test_model_loader.py`

**Success criteria:** `test_load_yolo_model` passes — loads .pt file and returns model object

**Steps:**

1. **Write test** `test_model_loader.py`:
   ```python
   import pytest
   from viz_tool.model_loader import load_yolo_model
   
   def test_load_yolo_model(tmp_path):
       """Test loading a YOLO model from .pt file."""
       # Use a mock .pt file or skip if no model available
       pytest.skip("Requires actual YOLO model file")
       
   def test_load_yolo_model_returns_ultralytics_model():
       """Verify returned object is ultralytics YOLO instance."""
       from ultralytics import YOLO
       # Mock test - will implement with real model path
       pytest.skip("Requires actual YOLO model file")
   ```

2. Verify test is skipped (expected: no model file yet)

3. **Implement** `model_loader.py`:
   ```python
   from pathlib import Path
   from ultralytics import YOLO
   from typing import Optional
   
   def load_yolo_model(model_path: str) -> Optional[YOLO]:
       """Load YOLO model from .pt file.
       
       Args:
           model_path: Path to .pt file
           
       Returns:
           YOLO model instance or None if file not found
       """
       path = Path(model_path)
       if not path.exists():
           return None
       return YOLO(str(path))
   ```

4. Update tests to use actual model paths when available

5. `git commit -m "feat: add model loader for YOLO .pt files"`

---

### Task 3: Grad-CAM Computation
**Files:** `viz_tool/gradcam.py`, `viz_tool/tests/test_gradcam.py`

**Success criteria:** `test_gradcam_output_shape` passes — returns heatmap of correct shape

**Steps:**

1. **Write test** `test_gradcam.py`:
   ```python
   import pytest
   import torch
   import numpy as np
   from viz_tool.gradcam import compute_gradcam
   
   def test_gradcam_output_shape():
       """Test Grad-CAM returns heatmap with correct shape."""
       # Mock model and input
       pytest.skip("Requires YOLO model with forward hooks")
       
   def test_gradcam_normalization():
       """Test heatmap values are normalized to [0, 1]."""
       pytest.skip("Requires implementation")
   ```

2. **Implement** `gradcam.py`:
   ```python
   import torch
   import numpy as np
   from typing import Tuple
   from ultralytics import YOLO
   
   class GradCAMExtractor:
       """Extract Grad-CAM heatmaps from YOLO model."""
       
       def __init__(self, model: YOLO, target_layer: str = "model.22"):
           self.model = model.model
           self.target_layer = self._get_layer(target_layer)
           self.gradients = None
           self.activations = None
           self._register_hooks()
       
       def _get_layer(self, layer_name: str):
           """Get layer by name."""
           for name, module in self.model.named_modules():
               if name == layer_name:
                   return module
           raise ValueError(f"Layer {layer_name} not found")
       
       def _register_hooks(self):
           """Register forward and backward hooks."""
           self.target_layer.register_forward_hook(self._save_activation)
           self.target_layer.register_backward_hook(self._save_gradient)
       
       def _save_activation(self, module, input, output):
           self.activations = output.detach()
       
       def _save_gradient(self, module, grad_input, grad_output):
           self.gradients = grad_output[0].detach()
       
       def compute_heatmap(self, image: np.ndarray, class_idx: int) -> np.ndarray:
           """Compute Grad-CAM heatmap for given image and class.
           
           Args:
               image: Input image (H, W, 3) normalized to [0, 1]
               class_idx: Target class index
               
           Returns:
               Heatmap (H, W) normalized to [0, 1]
           """
           # Forward pass
           input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
           output = self.model(input_tensor)
           
           # Backward pass
           self.model.zero_grad()
           output[0, class_idx].backward()
           
           # Compute Grad-CAM
           weights = self.gradients.mean(dim=(2, 3), keepdim=True)
           heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
           heatmap = torch.relu(heatmap)
           heatmap = heatmap.squeeze().cpu().numpy()
           
           # Normalize
           heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
           return heatmap
   
   def compute_gradcam(model: YOLO, image: np.ndarray, class_idx: int) -> np.ndarray:
       """Convenience function to compute Grad-CAM."""
       extractor = GradCAMExtractor(model)
       return extractor.compute_heatmap(image, class_idx)
   ```

3. Update tests with actual model and verify heatmap shape/normalization

4. `git commit -m "feat: add Grad-CAM computation for YOLO models"`

---

### Task 4: Feature Extractor
**Files:** `viz_tool/feature_extractor.py`, `viz_tool/tests/test_feature_extractor.py`

**Success criteria:** `test_extract_embeddings_shape` passes — returns (N, D) embeddings

**Steps:**

1. **Write test** `test_feature_extractor.py`:
   ```python
   import pytest
   import numpy as np
   
   def test_extract_embeddings_shape():
       """Test feature extraction returns correct shape."""
       pytest.skip("Requires YOLO model and images")
       
   def test_umap_projection():
       """Test UMAP projects to 3D."""
       pytest.skip("Requires implementation")
   ```

2. **Implement** `feature_extractor.py`:
   ```python
   import torch
   import numpy as np
   from typing import List, Tuple
   from ultralytics import YOLO
   from umap import UMAP
   
   class FeatureExtractor:
       """Extract embeddings from YOLO model."""
       
       def __init__(self, model: YOLO, layer_name: str = "model.21"):
           self.model = model.model
           self.layer = self._get_layer(layer_name)
           self.features = None
           self._register_hook()
       
       def _get_layer(self, layer_name: str):
           for name, module in self.model.named_modules():
               if name == layer_name:
                   return module
           raise ValueError(f"Layer {layer_name} not found")
       
       def _register_hook(self):
           self.layer.register_forward_hook(self._save_features)
       
       def _save_features(self, module, input, output):
           self.features = output.detach()
       
       def extract(self, images: List[np.ndarray]) -> np.ndarray:
           """Extract embeddings for batch of images.
           
           Args:
               images: List of images (H, W, 3)
               
           Returns:
               Embeddings (N, D) where D is embedding dimension
           """
           embeddings = []
           for img in images:
               input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
               _ = self.model(input_tensor)
               # Global average pooling
               feat = self.features.mean(dim=(2, 3)).squeeze().cpu().numpy()
               embeddings.append(feat)
           return np.stack(embeddings)
   
   def project_to_3d(embeddings: np.ndarray) -> np.ndarray:
       """Project embeddings to 3D using UMAP.
       
       Args:
           embeddings: (N, D) embeddings
           
       Returns:
           (N, 3) 3D coordinates
       """
       umap = UMAP(n_components=3, random_state=42)
       return umap.fit_transform(embeddings)
   ```

3. Update tests with actual data

4. `git commit -m "feat: add feature extractor with UMAP 3D projection"`

---

### Task 5: Visualization Module (Plotly Charts)
**Files:** `viz_tool/visualization.py`, `viz_tool/tests/test_visualization.py`

**Success criteria:** `test_create_3d_scatter` passes — returns Plotly figure

**Steps:**

1. **Write test** `test_visualization.py`:
   ```python
   import pytest
   import plotly.graph_objects as go
   import numpy as np
   
   def test_create_3d_scatter():
       """Test 3D scatter plot creation."""
       from viz_tool.visualization import create_3d_feature_space
       coords = np.random.randn(100, 3)
       labels = np.random.randint(0, 5, 100)
       fig = create_3d_feature_space(coords, labels)
       assert isinstance(fig, go.Figure)
       
   def test_create_gradcam_overlay():
       """Test Grad-CAM overlay visualization."""
       pytest.skip("Requires implementation")
   ```

2. **Implement** `visualization.py`:
   ```python
   import plotly.graph_objects as go
   import plotly.express as px
   import numpy as np
   from typing import List
   
   def create_3d_feature_space(
       coords: np.ndarray,
       labels: np.ndarray,
       class_names: List[str]
   ) -> go.Figure:
       """Create interactive 3D scatter plot of feature space.
       
       Args:
           coords: (N, 3) 3D coordinates
           labels: (N,) class labels
           class_names: List of class names
           
       Returns:
           Plotly Figure
       """
       fig = go.Figure()
       
       for class_idx in range(len(class_names)):
           mask = labels == class_idx
           fig.add_trace(go.Scatter3d(
               x=coords[mask, 0],
               y=coords[mask, 1],
               z=coords[mask, 2],
               mode='markers',
               marker=dict(size=4, opacity=0.7),
               name=class_names[class_idx]
           ))
       
       fig.update_layout(
           title="3D Feature Space",
           scene=dict(
               xaxis_title="UMAP-1",
               yaxis_title="UMAP-2",
               zaxis_title="UMAP-3"
           )
       )
       return fig
   
   def create_gradcam_overlay(
       image: np.ndarray,
       heatmap: np.ndarray,
       alpha: float = 0.5
   ) -> go.Figure:
       """Create Grad-CAM heatmap overlay.
       
       Args:
           image: Original image (H, W, 3)
           heatmap: Grad-CAM heatmap (H, W)
           alpha: Overlay opacity
           
       Returns:
           Plotly Figure
       """
       # Resize heatmap to image size
       from PIL import Image
       heatmap_resized = np.array(Image.fromarray(heatmap).resize(
           (image.shape[1], image.shape[0]), Image.BILINEAR
       ))
       
       fig = go.Figure()
       fig.add_trace(go.Image(z=image))
       fig.add_trace(go.Heatmap(
           z=heatmap_resized,
           colorscale='jet',
           opacity=alpha,
           showscale=False
       ))
       fig.update_layout(title="Grad-CAM Overlay")
       return fig
   
   def create_training_metrics(
       epochs: List[int],
       loss: List[float],
       map50: List[float]
   ) -> go.Figure:
       """Create training metrics dashboard.
       
       Args:
           epochs: List of epoch numbers
           loss: List of loss values
           map50: List of mAP@50 values
           
       Returns:
           Plotly Figure with subplots
       """
       from plotly.subplots import make_subplots
       
       fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "mAP@50"))
       
       fig.add_trace(go.Scatter(x=epochs, y=loss, name="Loss"), row=1, col=1)
       fig.add_trace(go.Scatter(x=epochs, y=map50, name="mAP@50"), row=1, col=2)
       
       fig.update_layout(title="Training Metrics", height=400)
       return fig
   ```

3. Run tests to verify figure creation

4. `git commit -m "feat: add Plotly visualization module"`

---

### Task 6: Utility Functions
**Files:** `viz_tool/utils.py`, `viz_tool/tests/test_utils.py`

**Success criteria:** `test_load_image_returns_correct_shape` and `test_get_image_paths_returns_list` pass

**Steps:**

1. **Write tests** `test_utils.py`:
   ```python
   import pytest
   import numpy as np
   from pathlib import Path
   from viz_tool.utils import load_image, get_image_paths
   
   def test_load_image_returns_correct_shape(tmp_path):
       """Test load_image returns (H, W, 3) array normalized to [0, 1]."""
       # Create dummy image
       import cv2
       img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
       img_path = tmp_path / "test.jpg"
       cv2.imwrite(str(img_path), img)
       
       result = load_image(str(img_path), target_size=640)
       assert result.shape == (640, 640, 3)
       assert result.dtype == np.float32
       assert result.min() >= 0.0 and result.max() <= 1.0
   
   def test_get_image_paths_returns_list(tmp_path):
       """Test get_image_paths returns list of image paths."""
       # Create dummy images
       import cv2
       for i in range(5):
           img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
           cv2.imwrite(str(tmp_path / f"img{i}.jpg"), img)
       
       paths = get_image_paths(str(tmp_path), max_images=3)
       assert isinstance(paths, list)
       assert len(paths) == 3
       assert all(p.endswith('.jpg') for p in paths)
   ```

2. **Implement** `utils.py`:
   ```python
   import cv2
   import numpy as np
   from pathlib import Path
   from typing import List
   
   def load_image(image_path: str, target_size: int = 640) -> np.ndarray:
       """Load and preprocess image for YOLO.
       
       Args:
           image_path: Path to image
           target_size: Target size (will be resized and padded)
           
       Returns:
           Preprocessed image (H, W, 3) normalized to [0, 1]
       """
       img = cv2.imread(image_path)
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
       # Resize with padding
       h, w = img.shape[:2]
       scale = target_size / max(h, w)
       new_h, new_w = int(h * scale), int(w * scale)
       img = cv2.resize(img, (new_w, new_h))
       
       # Pad
       pad_h = target_size - new_h
       pad_w = target_size - new_w
       img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
       
       # Normalize
       img = img.astype(np.float32) / 255.0
       return img
   
   def get_image_paths(dataset_path: str, max_images: int = 100) -> List[str]:
       """Get list of image paths from dataset.
       
       Args:
           dataset_path: Path to dataset directory
           max_images: Maximum number of images to return
           
       Returns:
           List of image paths
       """
       path = Path(dataset_path)
       extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
       images = [str(p) for p in path.rglob('*') if p.suffix.lower() in extensions]
       return images[:max_images]
   
   def load_training_logs(log_path: str) -> dict:
       """Load training logs from CSV file.
       
       Args:
           log_path: Path to results.csv
           
       Returns:
           Dictionary with epoch, loss, map50 lists
       """
       import pandas as pd
       df = pd.read_csv(log_path)
       return {
           'epochs': df['epoch'].tolist(),
           'loss': df['train/box_loss'].tolist(),
           'map50': df['metrics/mAP50(B)'].tolist()
       }
   ```

2. `git commit -m "feat: add utility functions for image loading and preprocessing"`

---

### Task 7: Main Streamlit App
**Files:** `viz_tool/app.py`

**Success criteria:** `streamlit run viz_tool/app.py` launches dashboard on localhost:8501

**Steps:**

1. **Implement** `app.py`:
   ```python
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
           st.write("Teacher model loaded successfully")
           st.write(f"Parameters: {sum(p.numel() for p in teacher_model.model.parameters()):,}")
       else:
           st.info("Load a teacher model to view architecture")
   
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
               teacher_extractor = GradCAMExtractor(teacher_model)
               teacher_heatmap = teacher_extractor.compute_heatmap(img, class_idx)
               teacher_fig = create_gradcam_overlay(img, teacher_heatmap)
               
               # Student Grad-CAM
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
           st.info("Load both models and dataset to compare Grad-CAM")
   
   # Tab 3: Feature Space 3D
   with tab3:
       st.header("3D Feature Space")
       
       if teacher_model and student_model and dataset_path:
           images = get_image_paths(dataset_path, max_images=100)
           if images:
               # Extract features
               img_batch = [load_image(p) for p in images]
               
               teacher_extractor = FeatureExtractor(teacher_model)
               teacher_embeddings = teacher_extractor.extract(img_batch)
               teacher_3d = project_to_3d(teacher_embeddings)
               
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
   ```

2. Test launch: `streamlit run viz_tool/app.py`

3. Verify app loads on localhost:8501

4. `git commit -m "feat: add main Streamlit app with 4 visualization tabs"`

---

### Task 8: Integration Test & Documentation
**Files:** `README-viz.md`

**Success criteria:** User can run app and see all 4 tabs working

**Steps:**

1. Create `README-viz.md`:
   ```markdown
   # Neural Network 3D Training Visualization
   
   Interactive Streamlit dashboard for visualizing YOLO model training.
   
   ## Features
   
   - **Architecture Viewer:** Inspect model structure and parameters
   - **Grad-CAM Comparison:** Side-by-side attention maps (teacher vs student)
   - **Feature Space 3D:** Interactive 3D embedding visualization with UMAP
   - **Training Metrics:** Real-time loss and mAP curves
   
   ## Quick Start
   
   ```bash
   # Install dependencies
   pip install -r requirements-viz.txt
   
   # Run the app
   streamlit run viz_tool/app.py
   ```
   
   Open http://localhost:8501 in your browser.
   
   ## Usage
   
   1. **Load Models:** Enter paths to teacher and student .pt files in sidebar
   2. **Load Dataset:** Enter path to validation image directory
   3. **Explore Tabs:** Navigate through Architecture, Grad-CAM, Feature Space, Metrics
   
   ## Requirements
   
   - Python 3.10+
   - GPU recommended (for Grad-CAM computation)
   - Trained YOLO models (.pt format)
   - Validation dataset (images)
   
   ## Next Steps
   
   This is an MVP. Future upgrades:
   - React Three Fiber for professional 3D
   - Real-time training updates via WebSocket
   - Multi-model comparison
   - Export animations
   ```

2. Test full workflow:
   - Launch app
   - Load dummy paths (app should show info messages)
   - Verify all 4 tabs render without errors

3. `git commit -m "docs: add README for visualization tool"`

4. `git commit -m "feat: complete Streamlit MVP for NN training visualization"`

---

## Execution Strategy

**TDD Approach:** Each task follows test-first pattern:
1. Write failing test
2. Implement minimal code to pass
3. Refactor if needed
4. Commit

**Time Estimate:** 2-3 hours total (30 min per task × 8 tasks, with buffer for debugging)

**Parallel Work:** Tasks 2-6 can be developed in parallel by different team members (see checklist).

---

## Verification

After all tasks complete:
```bash
# Install
pip install -r requirements-viz.txt

# Run
streamlit run viz_tool/app.py

# Open browser to http://localhost:8501
# Verify all 4 tabs load without errors
```
