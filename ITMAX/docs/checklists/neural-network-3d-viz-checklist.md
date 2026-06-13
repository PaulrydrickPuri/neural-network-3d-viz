# Work Package Checklist: Neural Network 3D Training Visualization (Streamlit MVP)

**Plan:** `docs/plans/neural-network-3d-viz.md`  
**Branch:** `feat/neural-network-3d-viz`  
**Status:** 🟢 Done  
**PR:** (pending)

---

## 🚀 DevOps / Setup
**Assignee:** autoship 🤖   **Status:** 🟢 Done
- [x] Task 1: Project setup & dependencies — `pip install -r requirements-viz.txt` succeeds

---

## 🧠 Backend (Python - Core Modules)
**Assignee:** autoship 🤖   **Status:** 🟢 Done
- [x] Task 2: Model loader module — `test_load_yolo_model` passes
- [x] Task 3: Grad-CAM computation — `test_gradcam_output_shape` passes
- [x] Task 4: Feature extractor with UMAP — `test_extract_embeddings_shape` passes
- [x] Task 5: Plotly visualization module — `test_create_3d_scatter` passes
- [x] Task 6: Utility functions (image loading, preprocessing) — `test_load_image_returns_correct_shape` and `test_get_image_paths_returns_list` pass

---

## 🎨 Frontend (Streamlit Dashboard)
**Assignee:** autoship 🤖   **Status:** 🟢 Done
- [x] Task 7: Main Streamlit app with 4 tabs — `streamlit run viz_tool/app.py` launches on localhost:8501

---

## 🧪 QA / Integration
**Assignee:** autoship 🤖   **Status:** 🟢 Done
- [x] Task 8: Integration test & documentation — app loads all 4 tabs without errors

---

## How to Pick Up a Work Package

### Step 1 — Claim it (edit this file)
```
**Assignee:** @your-name   **Status:** 🟡 In progress
```
Commit and push the change so teammates know it's taken.

### Step 2 — Create your worktree (run in project root)
```bash
git fetch origin
git worktree add -b feat/neural-network-3d-viz-[package] ../ITMAX-[package] origin/main
cd ../ITMAX-[package]
```

Replace `[package]` with your discipline: `backend`, `frontend`, `devops`, or `qa`.

### Step 3 — Open Claude Code and start work
```bash
claude
```

Then tell Claude:
```
I'm picking up the [Backend / Frontend / DevOps / QA] work package for neural-network-3d-viz.
My tasks are in docs/checklists/neural-network-3d-viz-checklist.md.
Let's start implementing.
```

Claude will read the plan and checklist and begin `/tdd` automatically.

### Step 4 — When all your tasks are done
```bash
/simplify   # always (reduces complexity, removes duplication)
/review     # always (self code-review before anyone else sees it)
/pr         # always (rebase, test, push, open PR, update checklist)
```

### Step 5 — After your PR is merged
```bash
cd ../ITMAX
git worktree remove ../ITMAX-[package]
git worktree prune
```

---

## Parallel Work Strategy

This project is designed for parallel execution:

1. **DevOps** (Task 1) must complete first — everyone needs dependencies
2. **Backend** (Tasks 2-6) can run in parallel — modules are independent
3. **Frontend** (Task 7) depends on Backend modules being ready
4. **QA** (Task 8) runs last — integration testing after everything works

**Recommended sequence for solo developer:**
- Task 1 → Tasks 2-6 (in order) → Task 7 → Task 8

**Recommended sequence for team:**
- DevOps: Task 1
- Backend Team: Tasks 2-6 (split among team members)
- Frontend: Task 7 (after Backend completes)
- QA: Task 8 (final integration)

---

## Quick Reference

**Total tasks:** 8  
**Estimated time:** 2-3 hours (solo) or 1 hour (team of 4 working in parallel)  
**Tech stack:** Python 3.10+ / Streamlit / Plotly / PyTorch / Ultralytics  
**Run command:** `streamlit run viz_tool/app.py`  
**Access URL:** http://localhost:8501
