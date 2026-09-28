# MCDM Decision Lab

Application Streamlit de sélection multicritère du meilleur modèle de Machine Learning pour la détection de fraude bancaire.

## Architecture

- `app.py` : programme principal
- `ui.py` : interface Streamlit et design
- `utils.py` : validations et fonctions utilitaires
- `methods/weighting.py` : Manual, AHP, BWM, Entropy, CRITIC
- `methods/normalization.py` : normalisations spécifiques aux méthodes
- `methods/ranking.py` : WSM, WPM, WASPAS, TOPSIS, VIKOR
- `data/model_selection.csv` : données d'exemple

## Installation

```bash
python -m pip install -r requirements.txt
```

## Lancement

```bash
python -m streamlit run app.py
```

## Scénario

Alternatives :
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- KNN

Critères :
- Precision (+)
- Recall (+)
- F1-score (+)
- AUPRC (+)
- Training time (-)
- Prediction time (-)
- Interpretability (+)
