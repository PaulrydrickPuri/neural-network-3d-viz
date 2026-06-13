# Neural Network 3D Training Visualization

Interactive Streamlit dashboard for visualizing YOLO model training with 3D feature space, Grad-CAM comparisons, and real-time metrics.

## Features

- **Architecture Viewer:** Inspect model structure and parameter counts
- **Grad-CAM Comparison:** Side-by-side attention maps (teacher vs student)
- **Feature Space 3D:** Interactive 3D embedding visualization with UMAP
- **Training Metrics:** Loss and mAP curves from training logs

## Quick Start

```bash
# Install dependencies
pip install -r requirements-viz.txt

# Run the app
streamlit run viz_tool/app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Load Models:** Enter paths to teacher and student `.pt` files in the sidebar
2. **Load Dataset:** Enter path to validation image directory
3. **Explore Tabs:**
   - **Architecture:** View model parameters
   - **Grad-CAM:** Compare where teacher vs student focus attention
   - **Feature Space 3D:** See how embeddings cluster by class
   - **Training Metrics:** Load `results.csv` to view training curves

## Requirements

- Python 3.10+
- GPU recommended (for Grad-CAM computation)
- Trained YOLO models (`.pt` format)
- Validation dataset (images)

## Example

```bash
# Run with your PPE models
streamlit run viz_tool/app.py

# In the browser:
# Teacher Model Path: /path/to/yolo26l_teacher.pt
# Student Model Path: /path/to/yolo26s_student.pt
# Dataset Path: /path/to/validation/images/
```

## Project Structure

```
viz_tool/
├── __init__.py
├── app.py                    # Main Streamlit dashboard
├── model_loader.py           # Load YOLO .pt files
├── gradcam.py                # Grad-CAM heatmap computation
├── feature_extractor.py      # Extract embeddings + UMAP
├── visualization.py          # Plotly chart builders
├── utils.py                  # Image loading utilities
└── tests/                    # Unit tests for each module
```

## Next Steps

This is an MVP. Future upgrades:
- React Three Fiber for professional 3D rendering
- Real-time training updates via WebSocket
- Multi-model comparison (beyond teacher/student)
- Export animations for presentations
- Class-specific Grad-CAM (select which class to visualize)

## Testing

Run the test suite:

```bash
python -m pytest viz_tool/tests/ -v
```

## License

