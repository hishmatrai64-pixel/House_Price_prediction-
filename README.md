# House Price Prediction: Supervised + Unsupervised ML with Streamlit

An end-to-end machine learning project on the Housing dataset (545 houses, 12 features). It predicts a house's price, its price category, and the segment it belongs to, and serves all three from a Streamlit app.

**Live app:** <add https://bfse8rllq3xkut4vrjubwd.streamlit.app/>

## What it does

| Task | Model | Result |
|---|---|---|
| Predict price (regression) | Linear regression in a scaling pipeline | about 63% cross-validated R² |
| Predict price category: Low / Medium / High / Very High (classification) | Logistic regression in a scaling pipeline | about 59% cross-validated accuracy |
| Group houses into segments (clustering) | K-means, k=4, in a scaling pipeline | four interpretable groups |

## Project steps

1. **Data cleaning:** checked missing values and duplicates, looked at outliers, encoded yes/no columns as 0/1, one-hot encoded furnishing status, and created `price_category` from price quartiles.
2. **Regression:** linear, Lasso, Ridge, decision tree, KNN, SVR and Random Forest, compared with train/test splits and 5-fold cross-validation.
3. **Classification:** logistic regression, Gaussian, Multinomial and Bernoulli Naive Bayes, decision tree, KNN, SVM and Random Forest, compared with accuracy, precision, recall, F1, confusion matrices and cross-validation.
4. **Clustering:** K-means with the elbow method and silhouette scores, then cluster profiling against price.
5. **Deployment:** the final models are saved with `joblib` and served by a Streamlit app.

## Key findings

- **Regression:** linear, Lasso and Ridge reach about 63% CV R² and are the best regressors. Random Forest (about 62%) and the scaled SVR (about 60%) follow. An unpruned decision tree overfits (train 99.9%, CV 11%). Tuning its depth and leaf size lifted CV R² to about 46%.
- **Lasso and Ridge** behave almost like plain linear regression, because the model is not overfitting, so regularization has little to correct.
- **Classification:** logistic regression and SVM tie at about 59% CV accuracy. Among the Naive Bayes variants, GaussianNB fits best, since the features are continuous. Multinomial and Bernoulli NB are designed for counts and binary data.
- **Clustering:** the four K-means segments are basic budget homes (average price about 3.4M), mid-range semi-furnished homes (4.3M), large homes with a basement and guestroom (5.8M) and premium multi-storey homes (6.8M). Although price was not used for clustering, the segments line up closely with price category.

## Limitations

- Small dataset (545 rows), so a single test split is noisy. Cross-validation is used for comparisons.
- R² of about 63% means individual predictions can be off by a million or more, and the most expensive houses are under-predicted.
- Silhouette scores are low (about 0.13 to 0.17), so the segments overlap and have soft boundaries.
- The data has no location or date, so the model cannot account for them.

## Run it locally

```bash
pip install -r requirements.txt
python train.py          # trains the models, creates house_models.joblib
streamlit run app.py
```

`train.py` expects the original `Housing.csv` (with `yes`/`no` text values) in the same folder.

## Files

| File | Purpose |
|---|---|
| `House_price_prediction_project.ipynb` | Full analysis: cleaning, modeling, evaluation, clustering |
| `train.py` | Trains the final pipelines and saves `house_models.joblib` |
| `app.py` | Streamlit app |
| `requirements.txt` | Dependencies |
| `Housing.csv` | Dataset |

## Tech

Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit, joblib.

## Author
<Hishmat Rai>
