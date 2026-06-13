# Neural Network 3D Training Visualization

**Date:** 2026-06-13  
**Status:** Decided → MVP First  
**Context:** Building interactive 3D visualization for neural network training (PPE detection, KD workflows)

---

## Problem Statement

Build an interactive 3D visualization that showcases neural network training (PPE detection or other use cases) running on localhost. The tool should visualize:
- 3D rotatable/zoomable neural network architecture
- Animated data flow through layers during training
- Teacher vs student comparison (for knowledge distillation)
- Feature space visualization (3D point cloud of embeddings)
- Grad-CAM heatmaps showing where model looks
- Real-time training metrics

**Constraints:**
- Must work with YOLO26 (PyTorch-based)
- Runs on localhost
- Should be interactive, not just pre-rendered video

---

## Options Considered

### Option A: Full Three.js + React from Scratch
- **How:** Custom 3D renderer with raw Three.js, React frontend, FastAPI backend
- **Pros:** Maximum control, professional-grade
- **Cons:** 2-4 days to prototype, requires WebGL knowledge, GPU contention
- **Complexity:** High

### Option B: React Three Fiber (R3F) ⭐ RECOMMENDED (Phase 2)
- **How:** React wrapper for Three.js. Declarative 3D with `<mesh>` components
- **Pros:** 50% less code than raw Three.js, React-friendly, professional output
- **Cons:** Still 1-2 days to prototype, learning curve
- **Complexity:** Medium-High

### Option C: Streamlit + Plotly 3D (MVP) ⭐ RECOMMENDED (Phase 1)
- **How:** Python-only. Streamlit for UI, Plotly for 3D plots, PyTorch hooks
- **Pros:** Working prototype in 2-3 hours, no frontend experience needed
- **Cons:** Less "video game" aesthetic, slower animations
- **Complexity:** Low

### Option D: Gradio + Embedded Three.js
- **How:** Gradio for ML interface, embed Three.js viewer
- **Pros:** Fast setup, ML-native
- **Cons:** Awkward integration, limited customization
- **Complexity:** Medium

### Option E: Manim (Pre-Rendered Animations)
- **How:** Python library for scripted animations (like 3Blue1Brown)
- **Pros:** Beautiful, perfect for papers/talks
- **Cons:** Not interactive, can't show live training
- **Complexity:** Medium (different skill set)

### Option F: Extend Existing Tool (Netron/TensorBoard)
- **How:** Add custom layers to Netron or TensorBoard Embedding Projector
- **Pros:** Start from working base, proven patterns
- **Cons:** Limited by existing architecture
- **Complexity:** Low-Medium

---

## Evaluation Matrix

| Option | Simplicity | Speed | Quality | Maintainability | Fit |
|--------|-----------|-------|---------|-----------------|-----|
| A: Raw Three.js | ❌ | ❌ | ✅ | ⚠️ | Only with 3D experience |
| B: React Three Fiber | ⚠️ | ⚠️ | ✅ | ✅ | Best balance |
| C: Streamlit + Plotly | ✅ | ✅ | ⚠️ | ✅ | **Best for validation** |
| D: Gradio + Three.js | ⚠️ | ⚠️ | ⚠️ | ⚠️ | Compromise |
| E: Manim | ⚠️ | ⚠️ | ✅ | ✅ | Presentations only |
| F: Extend existing | ✅ | ✅ | ⚠️ | ⚠️ | Quick win |

---

## Decision

**Two-phase approach:**

1. **Phase 1: Streamlit MVP (2-3 hours)**
   - Validate the core idea works
   - Show teacher/student comparison, feature space, Grad-CAM
   - If useful, proceed to Phase 2
   - If not, saved days of work

2. **Phase 2: React Three Fiber upgrade (1-2 days)**
   - Take validated Streamlit prototype
   - Rebuild with R3F for professional 3D
   - Know exactly what features matter

**Rationale:**
- Streamlit MVP validates the concept before committing to complex frontend
- React Three Fiber is the sweet spot between control and complexity
- Skipped raw Three.js (too complex without 3D experience)
- Skipped Manim (not interactive, user wants live training viz)

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU contention (3D + training both want GPU) | Run visualization on CPU or separate process |
| Streamlit too "basic" for final product | Accept MVP as validation, upgrade to R3F |
| Feature space too large to render | Use UMAP/t-SNE to project to 3D, sample 1000-5000 points |
| Real-time updates slow training | Update every N epochs (e.g., every 5 epochs) not every step |

---

## Next Steps

1. Create detailed implementation plan using `writing-plans` skill
2. Build Streamlit MVP with core features:
   - Model architecture viewer
   - Grad-CAM comparison (teacher vs student)
   - Feature space 3D visualization
   - Training metrics dashboard
3. Test on PPE training run
4. If useful, proceed to React Three Fiber upgrade
