# House price prediction with regression

Group project by four team members. Machine Learning 1, University Technical Degree in Artificial Intelligence (FCEIA, UNR, Argentina), second semester of 2026.

[Versión en español](README.md)

We predict `MEDV`, the median home value in thousands of dollars, from 13 variables in a 556-record dataset based on Boston Housing. We cleaned and imputed the data and compared seven models: linear regression, three gradient descent variants, Ridge, Lasso and ElasticNet. We chose linear regression, with an RMSE of 6.43 (thousands of dollars) and an R² of 0.62 on the test set.

## Team

This is a group project by four people:

- Valentino Civetta
- Maximiliano Frank
- Milagros Fucci
- Federico Winter

Original group repository: [valentinocivetta04/TP_AA1_Civetta_Fucci_Frank_Winter](https://github.com/valentinocivetta04/TP_AA1_Civetta_Fucci_Frank_Winter).

This repository is a copy under Maximiliano Frank's account, reorganized to show as a portfolio piece. The group owns the work. We worked together in Discord meetings, and whoever shared their screen made the commits. The git history therefore does not show who solved each part.

## My contribution

Like the rest of the group, I took part in every working session. We solved the notebook stages together: cleaning, exploratory analysis, imputation, modeling and conclusions.

## What we did

- We dropped 21 rows without `MEDV` and 4 rows with more than half of their values missing. That left 531 records.
- We marked invalid `RAD` values as missing.
- We detected outliers with the IQR rule and checked each one against the Spearman correlation matrix.
- We split train and test before scaling and imputing, and scaled with `RobustScaler`.
- We imputed continuous variables with `KNNImputer` (`k = 2`, chosen by the curvature of the elbow method). We imputed `RAD` and `CHAS` with `CatBoostClassifier`.
- We trained the seven models and tuned the learning rate, the batch size and `alpha` (by cross-validation).

## Key results

Test set metrics, sorted by RMSE. MAE and RMSE are in thousands of dollars.

| Model | Test MAE | Test RMSE | Test R² | Train RMSE |
|---|---|---|---|---|
| Mini-batch gradient descent | 4.1995 | 6.4244 | 0.6177 | 5.7154 |
| LinearRegression | 4.2010 | 6.4267 | 0.6174 | 5.7154 |
| Batch gradient descent | 4.2010 | 6.4267 | 0.6174 | 5.7154 |
| Stochastic gradient descent | 4.2127 | 6.4375 | 0.6161 | 5.7166 |
| ElasticNet | 4.2167 | 6.4774 | 0.6114 | 5.7602 |
| Lasso | 4.2354 | 6.4793 | 0.6112 | 5.7393 |
| Ridge | 4.2078 | 6.5089 | 0.6076 | 5.7791 |

- The seven models land within 0.1 RMSE of each other.
- Train and test are almost equal (R² of 0.61 and 0.62 for linear regression), so there is no overfitting. That is why regularization does not improve the metrics.
- Lasso sets the coefficients of `CRIM`, `NOX` and `RAD` to zero.
- Mini-batch beats `LinearRegression` by 0.002 RMSE. We attribute that gap to the method's oscillation and chose `LinearRegression`, which is exact and has no hyperparameters.

## Charts

![Distributions of the numeric variables](assets/01_distribuciones_variables.png)

*Distribution of the numeric variables, with mean and median (Figure 1 in the notebook).*

![Spearman correlation matrix](assets/02_correlacion_spearman.png)

*Spearman correlation matrix (Figure 5 in the notebook).*

![Gradient descent convergence](assets/03_convergencia_descenso_gradiente.png)

*Test MSE per epoch for the three gradient descent variants (Figure 13 in the notebook).*

![Regularization by alpha](assets/04_regularizacion_alpha.png)

*Ridge and Lasso coefficients and RMSE as a function of `alpha` (Figure 19 in the notebook).*

The charts are labeled in Spanish, as in the notebook.

## Technologies

Run with Python 3.13.16.

| Library | Version |
|---|---|
| pandas | 3.0.5 |
| numpy | 2.5.3 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| catboost | 1.2.10 |
| notebook | 7.6.3 |

## Repository structure

```
house_price_regression.ipynb   notebook with the full analysis and the models
data/house-prices-tp.csv       dataset provided by the course
assets/                        charts exported from the notebook
requirements.txt               libraries and versions
README.md                      Spanish README
README.en.md                   this file
```

## How to run it

The notebook reads the dataset from the relative path `./data/house-prices-tp.csv`, so open it from the repository root.

```bash
git clone https://github.com/MaxiFrank7/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook house_price_regression.ipynb
```

On Windows, the activation command is `.venv\Scripts\activate`.

The notebook text is in Spanish.

## Learnings and limitations

What we learned:

- Splitting train and test before scaling and imputing keeps the scaler and the imputers from using test information.
- The maximum learning rate for batch gradient descent can be computed from the data. In this problem it is 0.176.
- Regularization does not help when the model does not overfit.

Limitations:

- The model leaves about 40% of the price variability unexplained. On test it is off by 4,200 dollars (MAE) to 6,400 (RMSE).
- We think nonlinear relationships and the cap of 50 on `MEDV` play a role. We did not check this with a residual plot, so it remains a hypothesis.
- We did not try other train and test splits. With differences this small between models, the ranking could change.
- The dataset is small: 531 records after cleaning.
- The `B` variable in the original dataset is built from the racial composition of each area. scikit-learn removed `load_boston` in version 1.2 for that reason. We used the dataset because the course provided it.
- The gradient descent functions come from the course material. We used and compared them.
