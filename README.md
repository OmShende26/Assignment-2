## Files
- PCA_mushroom.ipynb: Full, reproducible workflow with code, plots, and narrative cells. It prints shapes after encoding, standardization status, scree plots, selected k, 2D PCA scatter, and classification reports for both models.[2][3]
- README.md (this file): How to run, design choices, and rubric-aligned justifications grounded in PCA and LR.

## Project overview
This assignment applies PCA to the UCI Mushroom dataset (categorical, high-dimensional, redundant features) and compares Logistic Regression performance on original vs PCA spaces. The pipeline includes one-hot encoding, standardization, PCA with scree analysis, 2D visualization, and performance comparison.

## Dataset
- Source: UCI Mushroom dataset (Agaricus-Lepiota). It has 8,124 samples, 22 categorical predictor attributes plus the target ‘class’ (e=edible, p=poisonous).
- All features are categorical; thus numeric encoders are required for most scikit-learn estimators.

## Environment and setup
- Dependencies: pandas, numpy, scikit-learn, matplotlib, seaborn.
- Running:
  - Open PCA_mushroom.ipynb.
  - Run all cells top-to-bottom; the notebook performs loading, encoding, scaling, PCA, plots, and classification.

## Part A: EDA and preprocessing
- One-hot encoding: Categorical features are mapped to a sparse/dense binary indicator matrix, one column per category, to embed categorical variables in a real vector space suitable for PCA and linear models. This is required so PCA can operate on vectors in ℝ^d and so Logistic Regression can consume numeric inputs.
- Dimensionality: After one-hot encoding, feature count expands substantially due to many categories per attribute; the notebook prints the post-encoding shape to show this blow-up.
- Standardization: Even though one-hot outputs are binary, centering to zero mean and scaling to unit variance equalizes variance across dummy columns and ensures PCA’s covariance-based components do not overweight high-frequency categories; scikit-learn supports scaling sparse/dense representations.

## Part B: PCA
- Fit PCA with no n_components initially to obtain the full explained_variance_ratio_ spectrum and cumulative sum. This yields the basis vectors (principal components) that maximize variance subject to orthogonality.
- Scree plot: The notebook plots explained variance ratio and cumulative explained variance vs component index, following standard scree methodology to identify elbows and target thresholds (e.g., 95%). Code mirrors common recipes for scree visualization.
- Selecting components: The notebook selects the minimal k achieving at least 95% cumulative variance (configurable), and justifies with the scree curve and the elbow heuristic. This balances information retention with dimension reduction.
- 2D visualization: Projects samples onto PC1–PC2 and colors by class to visually assess class separability in the reduced subspace; optional pair plots for additional PCs help gauge structure beyond 2D. This aligns with typical PCA visualization practice.

## Part C: Logistic Regression performance
- Baseline: Split standardized original one-hot features into train/test, train LogisticRegression (regularized, liblinear/lbfgs depending on sparsity/density), and report accuracy, precision, recall, and F1 via classification_report.
- PCA model: Transform train/test with the selected k PCs, train a new LogisticRegression on PC scores, and report the same metrics for fair comparison.
- Comparison and analysis:
  - When features are highly collinear and redundant (typical with one-hot expansions), PCA can decorrelate features and compress information, potentially improving generalization and training speed while slightly risking information loss depending on k.
  - Logistic Regression is an appropriate surrogate to quantify PCA effectiveness because it is sensitive to collinearity and dimensionality; improvements or stability in metrics after PCA suggest effective redundancy handling, while drops indicate harmful variance truncation.
