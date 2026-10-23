import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load():
    return joblib.load("house_models.joblib")

m = load()
lo, hi = m["area_range"]
st.title("House Price Predictor")
st.caption("Predicts price, price category and house segment from property features.")

c1, c2 = st.columns(2)
area = c1.number_input("Area (sq ft)", lo, hi, min(max(5000, lo), hi), step=100)
bedrooms = c1.slider("Bedrooms", 1, 6, 3)
bathrooms = c1.slider("Bathrooms", 1, 4, 1)
stories = c1.slider("Stories", 1, 4, 2)
parking = c1.slider("Parking spots", 0, 3, 1)
mainroad = int(c2.checkbox("Main road"))
guestroom = int(c2.checkbox("Guest room"))
basement = int(c2.checkbox("Basement"))
hotwater = int(c2.checkbox("Hot water heating"))
ac = int(c2.checkbox("Air conditioning"))
prefarea = int(c2.checkbox("Preferred area"))
furn = c2.selectbox("Furnishing", ["furnished", "semi-furnished", "unfurnished"])

row = pd.DataFrame([{
    "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories,
    "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
    "hotwaterheating": hotwater, "airconditioning": ac, "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus_semi-furnished": int(furn == "semi-furnished"),
    "furnishingstatus_unfurnished": int(furn == "unfurnished"),
}])[m["features"]]

if st.button("Predict", type="primary"):
    price = m["reg"].predict(row)[0]
    st.metric("Estimated price", f"{max(price, 0):,.0f}")
    st.write("**Price category:**", m["clf"].predict(row)[0])
    proba = pd.Series(m["clf"].predict_proba(row)[0], index=m["clf"].classes_)
    st.bar_chart(proba.reindex(["Low", "Medium", "High", "Very High"]))
    st.write("**House segment:**", m["segment_names"][int(m["seg"].predict(row)[0])])
