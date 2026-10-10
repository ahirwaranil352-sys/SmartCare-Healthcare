# AICTE × IBM SkillsBuild — Student Success AI

## AI Student Success Predictor & Early-Warning Dashboard

A runnable machine-learning project for the AICTE | IBM SkillsBuild Machine Learning & Applied AI internship.

The internship material identifies Masterclass 1 as “Set up VS Code & IBM BOB” and “Build Your First AI Model.” This project turns that learning into a stronger end-to-end portfolio project.

### Features
- Synthetic/demo student dataset generator
- scikit-learn Random Forest classifier
- Train/test evaluation
- Accuracy, precision, recall and F1
- Confusion matrix
- Feature importance
- Interactive Streamlit dashboard
- Individual student prediction
- Automated tests
- IBM Bob rules and AGENTS.md context
- No external dataset download required

> The dataset is synthetic/demo data. This project is for learning and demonstration, not for making real academic decisions.

## Run in VS Code

### 1. Create environment

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Generate data and train
```bash
python src/generate_data.py
python src/train_model.py
```

### 4. Start dashboard
```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.

### 5. Test
```bash
pytest -q
```

## IBM Bob workflow

Start in **Plan** mode:
```text
Review this project as an AI/ML mentor. Create a plan to improve data quality,
ML evaluation, explainability, UI usability and testing. Do not modify
application code yet. Keep the project beginner-friendly and runnable.
```

Then switch to **Agent** mode:
```text
Implement the approved improvement plan. Keep the existing run commands.
Do not add unnecessary dependencies. Keep synthetic data clearly labelled.
Update tests for changed behavior and run pytest -q after implementation.
Fix any actual failures and summarize every changed file.
```

Use **Ask** mode for viva preparation:
```text
Explain this project's complete ML pipeline like I am preparing for an
AICTE IBM SkillsBuild viva. Cover features, target, train/test split,
Random Forest, metrics, confusion matrix, probabilities, feature importance,
and limitations of synthetic data.
```

If Bob is not initialized for the project, run:
```text
/init
```

## Viva summary

**Problem:** Demonstrate how ML can classify student support-risk categories from structured academic and engagement features.

**Algorithm:** Random Forest classifier.

**Workflow:** data → validation → train/test split → model → evaluation → saved model → dashboard prediction.

**Important limitation:** The generated labels are based on a transparent synthetic rule. High accuracy on this dataset does not prove real-world validity.

## Future improvements
- Replace demo data with an approved real dataset.
- Compare Logistic Regression, Decision Tree and Random Forest.
- Add cross-validation and ROC-AUC.
- Add explainability such as SHAP.
- Add API deployment and Docker.
- Add privacy, fairness and drift checks before any real-world use.
