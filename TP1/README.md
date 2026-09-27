# TP1: predicción de precios de casas

Trabajo práctico 1 de Aprendizaje Automático 1, Tecnicatura Universitaria en Inteligencia Artificial (FCEIA, UNR).

Integrantes: Valentino Civetta, Milagros Fucci, Maximiliano Frank y Federico Winter.
Docentes: Giuliano Crenna, Tomás Avecilla y Agustín Alsop.

Predecimos `MEDV`, el valor mediano de las viviendas en miles de dólares, a partir de las otras 13 variables de un dataset de 556 registros basado en Boston Housing, usando regresión lineal múltiple.

En el notebook hacemos el análisis descriptivo y la limpieza de los datos, revisamos los outliers con IQR y su coherencia con la matriz de correlación, imputamos las variables continuas con `KNNImputer` y `RAD` y `CHAS` con `CatBoostClassifier`. Después comparamos `LinearRegression`, el descenso del gradiente en sus variantes batch, estocástico y mini-batch, Ridge, Lasso y ElasticNet. Nos quedamos con `LinearRegression`, que da un RMSE de 6.43 y un R² de 0.62 en test.

## Estructura

```
TP-regresion-AA1.ipynb     notebook principal
data/house-prices-tp.csv   dataset
docs/                      consigna e informe
src/                       notebooks auxiliares y pruebas previas
```

## Cómo correrlo

Lo trabajamos con Python 3.14. El notebook lee el dataset con una ruta relativa, así que hay que abrirlo desde esta carpeta:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scipy scikit-learn catboost jupyter
jupyter notebook TP-regresion-AA1.ipynb
```
