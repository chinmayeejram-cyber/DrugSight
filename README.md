# DrugSight - AI-assisted drug discovery

Predicts whether a molecule is **active against a disease target** (default: EGFR) with Random Forest, XGBoost
and a Graph Neural Network, **explains** the prediction atom-by-atom (SHAP), and screens **ADMET / drug-likeness**.

## Who owns what
| Folder | Owner | Files |
|---|---|---|
| `member1_cheminformatics/` | Member 1 | data_collection, data_cleaning, molecular_features -> **produces `data/processed/dataset.csv`** |
| `member2_machine_learning/` | Member 2 | featurize, random_forest, xgboost_model, gnn_model, model_evaluation, predictor, train_all |
| `admet/`, `explainability/` | Member 2 / 3 | drug_likeness.py, shap_analysis.py |
| `member3_web_application/` | Member 3 | app.py, dashboard.py, molecule_input.py |

## The ONE thing Member 1 must deliver
`data/processed/dataset.csv` with columns **`smiles`, `label`** (1 = active, e.g. pIC50 >= 6; 0 = inactive, pIC50 <= 5).
Optional columns: `molecule_chembl_id`, `pIC50`. Until it exists, everyone can work with `data/processed/demo_dataset.csv`
(fake toy data - never present results from it).

## Run order (from the project root, in PowerShell)
    pip install -r requirements.txt
    python -m member2_machine_learning.train_all          # trains RF + XGBoost + GCN, evaluates, makes SHAP plots
    streamlit run member3_web_application/app.py         # launches the web app

Train pieces separately:
    python -m member2_machine_learning.random_forest
    python -m member2_machine_learning.xgboost_model
    python -m member2_machine_learning.gnn_model --epochs 30
    python -m member2_machine_learning.model_evaluation
    python -m explainability.shap_analysis
    python -m admet.drug_likeness                        # ADMET demo on gefitinib

Options: `--csv path/to/file.csv`, `--skip-gnn` (train_all), `--epochs N`.

## Honest-science notes (say these in the presentation)
- Test set = molecules with **scaffolds never seen in training** (harder than a random split).
- ADMET values are **rule-based estimates** (Lipinski, Veber, Egan, ESOL, QED, PAINS/Brenk), not lab data.
- SHAP highlights what the *model* uses; it shows correlation in the data, not proven binding.

## Git tip
Add `results/models/` and `__pycache__/` to `.gitignore`, and rerun `train_all` on each laptop.

