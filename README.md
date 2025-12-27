# Single-Cell Perturbation Prediction Benchmarking

This repository provides end-to-end Jupyter notebooks for benchmarking **scGen**, **scPRAM**, and **CellOT** on single-cell perturbation prediction in two complementary settings:

- **Internal benchmarking on the Kang IFN-β dataset** (leave-one-cell-type-out within-study evaluation)
- **Cross-study extrapolation from Kang → Dong**, using both **scVI** and **Scanorama** integration to assess how batch correction influences perturbation prediction.

All workflows share a unified evaluation stack (R², Pearson, MSE, energy distance, MMD, Wasserstein distance, DEG Jaccard overlap, direction accuracy) at both global and cell-type resolution.

---

## Contents

- `scGen_benchmarking_workflow.ipynb`  
  Internal and cross-study benchmarking of **scGen**.
- `scPRAM_benchmarking_workflow.ipynb`  
  Internal and cross-study benchmarking of **scPRAM**.
- `cellOT_benchmarking_workflow.ipynb`  
  Internal and cross-study benchmarking of **CellOT / Mean Delta** baselines.
- `README.md`  
  This file.

You can treat each notebook as a standalone workflow, or run all three and use the cross-method comparison section to summarize results.

---

## Benchmarking Design

### 1. Internal Benchmarking (Kang dataset)

Internal evaluation is performed on the **Kang IFN-β PBMC dataset**, using a **leave-one-cell-type-out** scheme:

1. Integrate Kang data (or use as-is, depending on the method/section).
2. For each cell type:
   - Train on **control + IFN-β** cells from all other cell types.
   - Predict the held-out cell type’s stimulated (IFN-β) response from its control cells.
3. Compute metrics per cell type:
   - Mean-level metrics: **R²**, **Pearson**, **MSE**.
   - Distribution metrics: **energy distance**, **MMD**, **Wasserstein**, mean–variance KDE distance.
   - Biological metrics: **DEG Jaccard (top-100)** and **direction accuracy**.

This provides a controlled within-study measure of how each method extrapolates perturbation effects to unseen cell types.

### 2. Cross-Study Extrapolation (Kang → Dong)

Cross-study evaluation tests **true generalization** by predicting responses in an independent IFN-β dataset (Dong) from models trained on Kang:

1. Perform batch correction using either:
   - **scVI** integration, or
   - **Scanorama** integration.
2. Train each method on Kang (control + IFN-β) across common cell types.
3. Predict **Dong IFN-β** responses from **Dong control** cells.
4. Evaluate per cell type using the same metric stack as internal benchmarking.
5. Summarize:
   - Method performance: scGen vs scPRAM vs CellOT / Mean Delta.
   - Integration performance: scVI vs Scanorama (per method and on average).

This design allows you to disentangle:
- How good a **prediction model** is (scGen vs scPRAM vs CellOT), and  
- How much the **integration layer** (scVI vs Scanorama) limits or enables cross-study extrapolation.

---

## Notebooks in Detail

### `scGen_benchmarking_workflow.ipynb`

**Goal:** Evaluate scGen as a deep generative model for predicting single-cell perturbation responses.

**Key steps:**

- **Internal Kang benchmarking**
  - Apply leave-one-cell-type-out cross-validation on Kang.
  - Train scGen on control + IFN-β cells from all but one cell type.
  - Predict IFN-β response for held-out cell type from its control cells.
  - Compute per-cell-type metrics: R², Pearson, MSE, energy distance, MMD, Wasserstein, Jaccard, direction accuracy.
  - Inspect gene-level mean / variance R² to assess preservation of expression distributions.

- **Cross-study extrapolation (Kang → Dong)**
  - Use **scVI** and **Scanorama** to integrate Kang and Dong.
  - Train scGen on (integrated) Kang control + stimulated cells.
  - Predict Dong IFN-β responses from Dong controls in the integrated space.
  - Evaluate per cell type and compare integration strategies.

- **Integration validation**
  - UMAP/PCA colored by batch, dataset, cell type, and perturbation to verify batch mixing and label separation before prediction.

### `scPRAM_benchmarking_workflow.ipynb`

**Goal:** Evaluate scPRAM as a perturbation-aware deep generative model under the same protocols.

**Key steps:**

- **Internal Kang benchmarking**
  - Same leave-one-cell-type-out design, with scPRAM replacing scGen in the modeling step.
  - Full metric stack per cell type.

- **Cross-study extrapolation (Kang → Dong)**
  - Train scPRAM on Kang (scVI / Scanorama integrated).
  - Predict Dong IFN-β responses.
  - Compare scPRAM performance with scGen and CellOT for both integration strategies.

### `cellOT_benchmarking_workflow.ipynb`

**Goal:** Benchmark CellOT / Mean Delta-style baselines under the same data splits and integrations.

**Key steps:**

- **Internal Kang benchmarking**
  - Apply a mean-delta / OT-based mapping from control to stimulated.
  - Evaluate on held-out Kang cell types.

- **Cross-study extrapolation (Kang → Dong)**
  - Train mappings on Kang.
  - Apply to Dong controls (scVI / Scanorama integrated).
  - Compare as a baseline against scGen and scPRAM.
    
- **Comprehensive method comparisons**
  - Comparison: Internal Benchmarking - scGen vs scPRAM vs CellOT
  - Cross-Study Extrapolation: Integration Strategy Comparison
---

## Evaluation Metrics

For each method, integration, scenario, and cell type, the notebooks compute:

- **Mean-level accuracy**
  - **R²** between true and predicted mean expression.
  - **Pearson correlation** of gene-wise means.
  - **MSE** between mean expression profiles.

- **Distribution-level metrics**
  - **Energy distance** between real and predicted embeddings.
  - **MMD (RBF kernel)** between real and predicted expression distributions.
  - **Wasserstein distance** per gene, averaged across genes.

- **Biological relevance**
  - **DEG Jaccard (top-100)**: overlap of top differentially expressed genes between real and predicted vs. control.
  - **Direction Accuracy**: proportion of genes with correctly predicted sign of change.

Aggregated summaries (CSV + plots) are generated for:
- Internal benchmarking.
- Cross-study extrapolation.
- Integration comparison (scVI vs Scanorama).
- Method comparison (scGen vs scPRAM vs CellOT).

---

## Available Data

- **Kang dataset**: PBMC IFN-β stimulation with multiple annotated cell types.
- **Dong dataset**: Independent IFN-β perturbation dataset used as a cross-study target.
- Both datasets are preprocessed into `.h5ad` files with at least:
  - `obs['cell_type']`
  - `obs['perturbation']` (e.g. `ctrl`, `IFNb`)
  - `obs['batch']`
- Integration-specific notebooks assume access to:
  - **scVI-tools** for scVI integration.
  - **Scanorama** for Scanorama integration.
  - Standard Python scientific stack (NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, scanpy).

---

## Installation

Check each notebook’s first cell for the exact versions and dependencies.

---

## How to Run

1. **Clone the repository**
git clone https://github.com/katexiou/scGen-scPRAM-cellOT_benchmarking_workflows.git
cd scGen-scPRAM-cellOT_benchmarking_workflows


2. **Set up environment**
- Create and activate a virtualenv or conda env.
- Install the dependencies listed above.

3. **Data Availability**

Pre-computed results and processed datasets are available on Zenodo: [DOI: XXX/zenodo.XXXXXXX]

### Zenodo Contents

**Internal Benchmarking Results:**
- `scGen_all_cell_types_evaluation_results.csv` - scGen internal results
- `scPRAM_all_cell_types_evaluation_results.csv` - scPRAM internal results
- `cellot_results/cellot_results.csv` - CellOT internal results

**Cross-Study Extrapolation Results:**
- `scgen_scvi_integrated_cross_results.csv` - scGen + scVI
- `scgen_scanorama_integrated_cross_results.csv` - scGen + Scanorama
- `scpram_scvi_integrated_cross_results.csv` - scPRAM + scVI
- `scpram_scanorama_integrated_cross_results.csv` - scPRAM + Scanorama
- `cellot_cross_study_scvi/scvi_mean_delta_results.csv` - Mean Delta + scVI
- `cellot_cross_study_scanorama/scanorama_mean_delta_results.csv` - Mean Delta + Scanorama

**Processed Datasets:**
- `kang_processed.h5ad` - Preprocessed Kang PBMC dataset
- `kang_final.h5ad` - Final Kang dataset for benchmarking
- `Kang_Dong_integrated_filtered_final.h5ad` - scVI-integrated Kang-Dong dataset
- `unintegrated_Kang_Dong.h5ad` - Unintegrated merged Kang-Dong dataset

These files allow users to reproduce the final comparison and visualizations without rerunning the full training pipelines.

4. **Run notebooks in order**
- `scGen_benchmarking_workflow.ipynb`
- `scPRAM_benchmarking_workflow.ipynb`
- `cellOT_benchmarking_workflow.ipynb`

5. **Inspect outputs**
- CSVs
- Plots
- Summary CSVs for:
  - Method comparison.
  - Integration comparison.
  - Per-cell-type metrics.

---

## Output Files (csv)

### Individual Method Results
- `scGen_all_cell_types_evaluation_results.csv`
- `scgen_scvi_integrated_cross_results.csv`
- `scgen_scanorama_integrated_cross_results.csv`
- `scPRAM_all_cell_types_evaluation_results.csv`
- `scpram_scvi_integrated_cross_results.csv`
- `scpram_scanorama_integrated_cross_results.csv`
- `./cellot_results/cellot_results.csv`
- `./cellot_cross_study_scvi/scvi_mean_delta_results.csv`
- `./cellot_cross_study_scanorama/scanorama_mean_delta_results.csv`

### Cross-Method Comparison
- `cross_study_summary.csv`
- `cross_study_integration_comparison.csv`
- `cross_study_prediction_comparison.csv`
- `cross_study_detailed_comparison.csv`
- `cross_study_rankings.csv`

Plus multiple PNGs summarizing visualizations.

---

## Citation

If you use these workflows, please cite the original methods and datasets:

- Lotfollahi et al., scGen: Generative modeling and latent space arithmetic for single-cell perturbation responses.
- Lopez et al., scVI: A deep generative model for single-cell transcriptomics.
- Hie et al., Scanorama: Efficient integration of heterogeneous single-cell transcriptomes.
- Kang et al., Multiplexed droplet single-cell RNA-sequencing of stimulated and unstimulated PBMCs.
- Dong et al., IFN-β single-cell PBMC study (or the exact reference you use).

---

## License

MIT License – see LICENSE for details.

---

## Contact

For questions or suggestions, please open an issue or contact:

- `Aikaterini Alexiou` – `katerina.alexiou98@gmail.com`
  
