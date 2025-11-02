
# FlexTrack 2025 Demand Response Prediction Pipeline

## Overview
This repository contains a high-performance Python pipeline for the **FlexTrack Challenge 2025**, focused on detecting and quantifying energy flexibility in buildings via demand response (DR) events. The solution implements a **hybrid two-stage ensemble framework**:

- **Stage 1**: Classifies DR events (Demand Response Flag: -1 for decrease, 0 for normal, +1 for increase) using LightGBM and XGBoost ensembles.
- **Stage 2**: Regresses the energy impact (Demand Response Capacity in kW) during detected events, conditioned on the classification output.

Key innovations include:
- Data-driven site archetyping (small/large buildings based on mean power consumption).
- Extensive feature engineering (>100 features capturing temporal, weather, and operational dynamics).
- Hybrid ensembling of global and archetype-specific models for robust generalization.

The pipeline was developed for the FlexTrack Challenge and achieved a **G-Mean of 0.618** (classification) and **nMAE of 0.991** (regression) on the private test set.

For a detailed academic description, see the accompanying paper: **[A Hybrid Two-Stage Ensemble Framework for Detecting and Quantifying Energy Flexibility in Buildings: A Solution to the FlexTrack Challenge](paper.pdf)**.

Kaggle Notebook: **[FlexTrack 2025 Hybrid Ensemble Solution](https://www.kaggle.com/code/dagloxkankwanda/flextrack-aicrowd-notebook/edit)**

### Architecture Summary
![Model Architecture](/Model%20Architecture.jpg)
*(High-level pipeline: Feature engineering → Parallel global/archetype models → Weighted ensemble.)*

## Requirements
- Python 3.8+ (tested on 3.12).
- Dependencies: Installed via pip (see below).

## Installation
1. Clone or download this repository.
2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn scipy lightgbm xgboost
   ```
   - **Note**: For GPU acceleration (optional, faster training), ensure CUDA is installed and use `device="gpu"` in the script (default is CPU for local compatibility).

## Data Requirements
Place the following CSV files in the project root directory (or update paths in `flextrack_pipeline.py`):
- `flextrack-2025-training-data-v0.2.csv`: Training data (sites A-M, with ground truth flags and capacities).
- `flextrack-2025-public-test-data-v0.3.csv`: Test data (unlabeled, for predictions).
- `flextrack-2025-random-prediction-data-v0.2.csv`: Sample submission (optional, for format reference).

These files are available from the [FlexTrack Challenge on AICrowd](https://www.aicrowd.com/challenges/flextrack-challenge-2025/dataset_files).

## Usage
1. Ensure data files are in place.
2. Run the pipeline:
   ```
   python flextrack_pipeline.py
   ```
   - **Output**: Console progress logs + `submission.csv` (sorted by Site and Timestamp_Local, with predicted `Demand_Response_Flag` and `Demand_Response_Capacity_kW`).
   - **Runtime**: ~10-20 minutes on a standard CPU (longer for first run due to feature engineering; scales with data size).
   - **Customization**:
     - Set `RUN_GRID_SEARCH = True` for hyperparameter tuning (slower).
     - Set `USE_CALIBRATION = True` for isotonic regression on capacities (requires validation data).
     - Adjust ensemble weights (`global_weight = 0.9`, `site_weight = 0.1`) in the script.

### Example Output
The script prints progress like:
```
Loading data...
Generated Site Archetypes: {'SiteA': 'small', 'SiteB': 'large', ...}
...
--- STEP 1: PROCESSING GLOBAL MODEL ---
  - Global model using 45 clf features and 85 reg features.
...
Ensembling submission file 'submission.csv' created successfully.
```

### Sample Submission Format
`submission.csv` matches the sample:
```
Site,Timestamp_Local,Demand_Response_Flag,Demand_Response_Capacity_kW
SiteA,2025-01-01 00:00:00,-1,15.2
SiteA,2025-01-01 00:15:00,0,0.0
...
```

## Evaluation
- **Metrics** (as per challenge):
  - Classification: Geometric Mean (G-Mean), Macro F1-Score.
  - Regression: Normalized MAE (nMAE), Normalized RMSE (nRMSE).
- To evaluate locally: Split training data temporally and compute metrics using `sklearn.metrics`.

See the paper for ablation studies and detailed results (e.g., Round 1 vs. Round 2 performance).

## Project Structure
```
flextrack_pipeline/
├── data/
|   ├── flextrack-2025-public-test-data-v0.2.csv 
|   ├── flextrack-2025-public-test-data-v0.3.csv
|   ├── flextrack-2025-random-prediction-data-v0.2.csv
|   └── flextrack-2025-training-data-v0.2.csv
├── flextrack_pipeline.py     
├── flextrack-aicrowd-notebook.ipynb                 
├── LICENSE                 
├── README.md                 
├── Paper.pdf                 
├── Model Architecture.png   
└── submission.csv            
```

## Limitations and Future Work
- **Archetyping**: Binary (small/large); could extend to k-means clustering on load profiles.
- **Scalability**: CPU fallback for local runs; GPU recommended for large datasets.
- **Enhancements**: Add Transformer models for long-range dependencies or online learning for streaming data.
- See the paper's Discussion section for more.

## License
MIT License. See [LICENSE](LICENSE.txt).

## Acknowledgments
- Developed for the FlexTrack Challenge 2025 by **Daglox Kankwanda** (dagloxkankwanda@gmail.com).
- Thanks to AICrowd, NSW Government, CSIRO, and RACE for 2030 CRC for the dataset and platform.
- Code inspired by building energy ML literature (citations in paper).

## Citation
If using this work, please cite the paper:
```
@INPROCEEDINGS{kankwanda2025flextrack,
  author={Kankwanda, Daglox},
  title={A Hybrid Two-Stage Ensemble Framework for Detecting and Quantifying Energy Flexibility in Buildings: A Solution to the FlexTrack Challenge},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  doi={...}
}
```
