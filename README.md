# Data Science Exercises (Python)

This repository contains a collection of university exercises and exam practice material for Data Science, mainly developed with Python notebooks.

The project includes:
- standalone notebooks on core topics (PCA, data loading, general exercises),
- grouped exercises on binary classification and stochastic gradient descent (SGD),
- recap exercises with related datasets,
- exam simulation work and final test notebooks.

## Repository Structure

```text
py/
├── 04.12.2025.ipynb
├── load_wine.ipynb
├── pca_examples.ipynb
├── esame26Feb2026.zip
├── BinaryClassification&SGD/
│   ├── ExerciseOnBinaryClassification.ipynb
│   ├── ExerciseOnTheStochasticGradientDescentAlgorithmForSupervisedClassification.ipynb
│   ├── esercizio SGD.pdf
│   └── exercises_classification_and_SGD.pdf
├── exerciseRiepilogativi/
│   ├── ese1.ipynb
│   ├── ese2.ipynb
│   ├── ese3.ipynb
│   ├── exercise_21_01_26.npy
│   ├── online_learning_data.npy
│   └── tracce_esercizi_ricapitolativi_ef2abd40447b0469214d051c31c8b3cf.pdf
├── esame26Feb2026/
│   ├── esame26Feb2026.ipynb
│   └── data_sets/
│       ├── exercise1_26_02_26.csv
│       ├── exercise2_26_02_26_data_set_PART1.npy
│       └── exercise2_26_02_26_data_set_PART2.npy
└── prova_finale/
    ├── prova_finale.ipynb
    ├── eseClassificatore.pdf
    └── exercise_21_01_26.npy
```

## Topics Covered

- Binary classification workflows
- Stochastic Gradient Descent (SGD)
- Principal Component Analysis (PCA)
- Dataset handling with CSV and NumPy arrays
- Exam-style machine learning exercises

## Requirements

Recommended environment:
- Python 3.10+
- Jupyter Notebook or JupyterLab

Common Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Install dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Use

1. Open the project folder in your IDE.
2. Launch Jupyter:

   ```bash
   jupyter lab
   ```

   or

   ```bash
   jupyter notebook
   ```

3. Navigate to the notebook you want to run.
4. Execute cells in order to reproduce outputs.

## Notes

- Some exercises use local dataset files (`.csv`, `.npy`) stored next to the notebooks.
- Keep folder structure unchanged to avoid broken relative paths.
- PDF files provide exercise statements and supporting material.
