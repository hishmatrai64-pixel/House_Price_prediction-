"""Train and save the models. Run once:  python train.py
Expects Housing.csv (the ORIGINAL dataset with text yes/no columns) in the same folder."""
import joblib, pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Housing.csv")
for c in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]:
    df[c] = df[c].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True, dtype=int)

df["price_category"] = pd.qcut(df["price"], q=4, labels=["Low","Medium","High","Very High"])
X = df.drop(columns=["price","price_category"])
features = X.columns.tolist()

reg = make_pipeline(StandardScaler(), LinearRegression()).fit(X, df["price"])
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, df["price_category"])
seg = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42, n_init=10)).fit(X)

# name clusters by average price (cheapest -> most expensive), so names never depend on cluster ids
order = df.groupby(seg.predict(X))["price"].mean().sort_values().index.tolist()
names = ["Basic budget homes", "Mid-range semi-furnished homes", "Large homes with basement and guestroom", "Premium multi-storey homes"]
segment_names = {int(c): n for c, n in zip(order, names)}

joblib.dump({"reg": reg, "clf": clf, "seg": seg, "features": features,
             "segment_names": segment_names,
             "area_range": (int(df.area.min()), int(df.area.max()))}, "house_models.joblib")
print("saved", features, segment_names)
