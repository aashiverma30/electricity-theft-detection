import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="VoltGuard AI", page_icon="⚡", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: white;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.title-text {
    font-size: 45px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg,#FFD700,#FFFFFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

[data-testid="stSidebar"] {
    background: #050505;
}

.card {
    background: linear-gradient(145deg,#111111,#1a1a1a);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(255,215,0,0.2);
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-text'>⚡ VoltGuard AI</div>", unsafe_allow_html=True)
st.markdown("<center>Advanced Electricity Theft Monitoring System</center>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD DATA ----------------
try:
    @st.cache_data
    def load_data():
        return pd.read_csv("electricity_theft_dataset_3000houses.csv")

    df = load_data()
except:
    st.error("Dataset not found")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚡ VoltGuard Control")

menu = st.sidebar.radio("", [
    "📊 Dashboard",
    "🏠 House Monitoring",
    "🚨 Theft Alerts",
    "👨‍💼 Verification",
    "➕ Add New House"
])

# month = st.sidebar.selectbox("Select Month", df["Month"].unique())
# df = df[df["Month"] == month]

month = st.sidebar.selectbox("Select Month", df["Month"].unique())

filtered_df = df[df["Month"] == month]

# ---------------- FEATURE ENGINEERING ----------------
df["Difference"] = df["Current_Units"] - df["Average_Units"]
df["Deviation_%"] = ((df["Current_Units"] - df["Average_Units"]) / df["Average_Units"]) * 100
# 🔥 NEW SMART FEATURES
df["Usage_Ratio"] = df["Current_Units"] / df["Average_Units"].replace(0, 1)
df.replace([float("inf"), -float("inf")], 0, inplace=True)

df["High_Usage_Flag"] = df["Current_Units"].apply(lambda x: 1 if x > 400 else 0)

df["Low_Usage_Flag"] = df["Current_Units"].apply(lambda x: 1 if x < 100 else 0)
df["Target"] = df.apply(
    lambda row: 1 if (row["Deviation_%"] > 45 or row["Deviation_%"] < -30 or row["Usage_Ratio"] > 1.6)
    else 0,
    axis=1
)

le = LabelEncoder()
df["Month_Encoded"] = le.fit_transform(df["Month"])

# 🔥 AI EXPLANATION FUNCTION
def get_reasons(deviation, usage_ratio, high_flag, low_flag):
    reasons = []

    if deviation > 50:
        reasons.append("Unusually high consumption")

    if deviation < -35:
        reasons.append("Sudden drop in usage")

    if usage_ratio > 1.5:
        reasons.append("Usage ratio too high")

    if high_flag == 1:
        reasons.append("Very high usage detected")

    if low_flag == 1:
        reasons.append("Very low usage (possible tampering)")

    return ", ".join(reasons) if reasons else "Normal usage"

# ---------------- MODEL ----------------
X = df[[
    "Current_Units",
    "Average_Units",
    "Month_Encoded",
    "Deviation_%",
    "Difference",
    "Usage_Ratio",
    "High_Usage_Flag",
    "Low_Usage_Flag"
]]
y = df["Target"]

@st.cache_resource
def load_model(df):

    X = df[[
        "Current_Units",
        "Average_Units",
        "Month_Encoded",
        "Deviation_%",
        "Difference",
        "Usage_Ratio",
        "High_Usage_Flag",
        "Low_Usage_Flag"
    ]]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X, y)
    return model

model = load_model(df)

# # ---------------- ACCURACY ----------------
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.3,
#     random_state=42
# )

# from sklearn.ensemble import RandomForestClassifier

# temp_model = RandomForestClassifier(
#     n_estimators=20,
#     max_depth=3,
#     random_state=42
# )

# temp_model.fit(X_train, y_train)

# predictions = temp_model.predict(X_test)

# accuracy = accuracy_score(y_test, predictions)

# st.sidebar.success("Model Accuracy: 97.95%")
# accuracy = accuracy_score(y_test,model.predict(X_test))
# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, model.predict(X_test))

# st.write("Confusion Matrix:")
# st.write(cm)

df["Prediction"] = model.predict(X)
df["Theft_Status"] = df["Prediction"].apply(lambda x:"Suspicious" if x==1 else "Normal")
df["AI_Reason"] = df.apply(
    lambda row: get_reasons(
        row["Deviation_%"],
        row["Usage_Ratio"],
        row["High_Usage_Flag"],
        row["Low_Usage_Flag"]
    ),
    axis=1
)
# df.to_csv("electricity_theft_dataset_3000houses.csv", index=False)

# ---------------- DASHBOARD ----------------
# ---------------- DASHBOARD ----------------
if menu == "📊 Dashboard":

    # ---------------- DASHBOARD ----------------
    total = len(filtered_df)
    suspicious = len(filtered_df[filtered_df["Theft_Status"] == "Suspicious"])
    normal = total - suspicious

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Houses", total)
    col2.metric("Suspicious ⚡", suspicious)
    col3.metric("Normal", normal)

    if suspicious > total * 0.3:
        st.error("🚨 ALERT: Too many suspicious houses!")

    # 📊 Charts
    small_df = filtered_df.head(100)   # 👈 limit data

    st.line_chart(small_df.set_index("House_ID")["Current_Units"])
    st.bar_chart(small_df.set_index("House_ID")[["Current_Units", "Average_Units"]])
    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### 🔍 Feature Importance (AI Insights)")

    importance = model.feature_importances_
    features = X.columns

    sorted_idx = importance.argsort()
    sorted_features = features[sorted_idx]
    sorted_importance = importance[sorted_idx]

    imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

    st.bar_chart(imp_df.set_index("Feature"))

    # ================= ONLY DASHBOARD ME CHECK =================
    st.markdown("---")
    st.markdown("## 🔍 Check Electricity Theft")

    current_units = st.number_input("Current Units", 0, key="dashboard_current")
    avg_units = st.number_input("Average Units", 1, key="dashboard_avg")
    month_input = st.selectbox("Month Input", df["Month"].unique(), key="dashboard_month")

    if st.button("Analyze Usage"):

        deviation = ((current_units - avg_units) / avg_units) * 100
        month_encoded = le.transform([month_input])[0]

        difference = current_units - avg_units
        usage_ratio = current_units / avg_units

        high_flag = 1 if current_units > 400 else 0
        low_flag = 1 if current_units < 100 else 0

        input_df = pd.DataFrame({
            "Current_Units": [current_units],
            "Average_Units": [avg_units],
            "Month_Encoded": [month_encoded],
            "Deviation_%": [deviation],
            "Difference": [difference],
            "Usage_Ratio": [usage_ratio],
            "High_Usage_Flag": [high_flag],
            "Low_Usage_Flag": [low_flag]
        })

        pred = model.predict(input_df)[0]
        confidence = max(model.predict_proba(input_df)[0]) * 100
        reason = get_reasons(deviation, usage_ratio, high_flag, low_flag)

        st.markdown("### 🧠 AI Explanation")
        st.info(reason)

        if pred == 1:
            st.error(f"⚠️ Theft Risk HIGH ({round(deviation,2)}%) | Confidence: {round(confidence,2)}%")
        else:
            st.success(f"✅ Normal Usage ({round(deviation,2)}%)")

            # ---------------- GOVERNMENT IMPACT PANEL ----------------
    st.markdown("## 🇮🇳 Government & DISCOM Impact Analysis")

    estimated_loss = suspicious * 5000  # hypothetical revenue loss per case

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>Estimated Revenue Risk</h3>
            <h2>₹ {estimated_loss}</h2>
            <p>Potential loss due to suspicious activity</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h3>Grid Stability Impact</h3>
            <h2>{suspicious} Zones</h2>
            <p>Regions requiring inspection & load balancing</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "VoltGuard supports Digital India & Smart Grid initiatives by "
        "reducing electricity theft, improving transparency, and enabling "
        "data-driven enforcement for DISCOM authorities."
    )
   
# ---------------- HOUSE ----------------
elif menu == "🏠 House Monitoring":

    house_id = st.selectbox("Select House ID", df["House_ID"].unique())
    house = df[df["House_ID"] == house_id]

    # -------- CURRENT DATA --------
    st.subheader("📊 Current Data")
    st.dataframe(house)

    if house["Theft_Status"].values[0] == "Suspicious":
        st.error("⚡ Abnormal Usage")
    else:
        st.success("Normal Usage")

    # -------- PAST HISTORY --------
    st.markdown("---")
    st.subheader("📜 Past Verification History")

    try:
        history = pd.read_csv("verification_log.csv")

        # Same house ka history filter
        house_history = history[history["House_ID"] == house_id]

        if house_history.empty:
            st.info("No past records found for this house")
        else:
            st.dataframe(house_history)

    except:
        st.info("No verification history available")
# ---------------- ALERTS ----------------
elif menu == "🚨 Theft Alerts":

    alerts = df[df["Theft_Status"]=="Suspicious"]

    if alerts.empty:
        st.success("No Theft Detected")
    else:
        st.dataframe(alerts)

# ---------------- VERIFICATION ----------------
# ---------------- VERIFICATION ----------------
elif menu == "👨‍💼 Verification":

    st.subheader("Inspector Panel")

    suspicious_ids = df[df["Theft_Status"]=="Suspicious"]["House_ID"].unique()

    if len(suspicious_ids) > 0:

        house_id = st.selectbox("Select House", suspicious_ids)

        decision = st.radio("Decision", ["Confirm Theft","False Alarm","Under Investigation"])
        remarks = st.text_area("Remarks")

        if st.button("Submit"):

            new = pd.DataFrame({
                "House_ID":[house_id],
                "Month":[month],
                "Decision":[decision],
                "Remarks":[remarks],
                "Time":[datetime.now()]
            })

            try:
                old = pd.read_csv("verification_log.csv")
                updated = pd.concat([old,new],ignore_index=True)
            except:
                updated = new

            # ✅ Save log
            updated.to_csv("verification_log.csv", index=False)

            # 🔥 UPDATE MAIN DATASET
            main_df = pd.read_csv("electricity_theft_dataset_3000houses.csv")

            if decision == "Confirm Theft":
                main_df.loc[
                    (main_df["House_ID"] == house_id) & (main_df["Month"] == month),
                    "Target"
                ] = 1

            elif decision == "False Alarm":
                main_df.loc[
                    (main_df["House_ID"] == house_id) & (main_df["Month"] == month),
                    "Target"
                ] = 0

            # 🔥 Recalculate features after update
            main_df["Difference"] = main_df["Current_Units"] - main_df["Average_Units"]
            main_df["Deviation_%"] = ((main_df["Current_Units"] - main_df["Average_Units"]) / main_df["Average_Units"]) * 100
            main_df["Usage_Ratio"] = main_df["Current_Units"] / main_df["Average_Units"]
            main_df["High_Usage_Flag"] = main_df["Current_Units"].apply(lambda x: 1 if x > 400 else 0)
            main_df["Low_Usage_Flag"] = main_df["Current_Units"].apply(lambda x: 1 if x < 100 else 0)

            # 🔥 Re-generate AI Reason
            main_df["AI_Reason"] = main_df.apply(
                lambda row: get_reasons(
                    row["Deviation_%"],
                    row["Usage_Ratio"],
                    row["High_Usage_Flag"],
                    row["Low_Usage_Flag"]
                ),
                axis=1
            )

            main_df.to_csv("electricity_theft_dataset_3000houses.csv", index=False)
            st.success("Saved + Dataset Updated ✅")

            # 🔥 AUTO REFRESH
            st.rerun()

    else:
        st.info("No suspicious houses")

    # # -------- HISTORY --------
    # st.markdown("---")
    # st.subheader("📜 History")

    # try:
    #     history = pd.read_csv("verification_log.csv")
    #     st.dataframe(history)
    # except:
    #     st.info("No history found")

    # -------- HISTORY + DELETE --------
    st.markdown("---")
    st.subheader("📜 History")

    try:
        history = pd.read_csv("verification_log.csv")
        st.dataframe(history)

        idx = st.selectbox("Select record to delete", history.index)

        if st.button("Delete Record"):
            history = history.drop(idx)
            history.to_csv("verification_log.csv",index=False)
            st.success("Deleted")
            st.experimental_rerun()

    except:
        st.info("No history found")
# ---------------- ADD NEW HOUSE ----------------
elif menu == "➕ Add New House":

    st.subheader("➕ Register New House")

    new_house_id = st.number_input("House ID", min_value=1, key="new_house_id")
    
    new_month = st.selectbox("Month", df["Month"].unique(), key="new_month")

    new_current_units = st.number_input("Current Units", min_value=0, key="new_current")
    
    new_avg_units = st.number_input("Average Units", min_value=1, key="new_avg")

    if st.button("Add House"):

        deviation = ((new_current_units - new_avg_units) / new_avg_units) * 100
        month_encoded = le.transform([new_month])[0]

        # features
        difference = new_current_units - new_avg_units
        usage_ratio = new_current_units / new_avg_units

        high_flag = 1 if new_current_units > 400 else 0
        low_flag = 1 if new_current_units < 100 else 0

        input_df = pd.DataFrame({
            "Current_Units":[new_current_units],
            "Average_Units":[new_avg_units],
            "Month_Encoded":[month_encoded],
            "Deviation_%":[deviation],
            "Difference":[difference],
            "Usage_Ratio":[usage_ratio],
            "High_Usage_Flag":[high_flag],
            "Low_Usage_Flag":[low_flag]
        })

        pred = model.predict(input_df)[0]
        status = "Suspicious" if pred == 1 else "Normal"

        reason = get_reasons(deviation, usage_ratio, high_flag, low_flag)

        new_data = pd.DataFrame({
            "House_ID":[new_house_id],
            "Month":[new_month],
            "Current_Units":[new_current_units],
            "Average_Units":[new_avg_units],
            "Difference":[difference],
            "Deviation_%":[deviation],
            "Usage_Ratio":[usage_ratio],
            "High_Usage_Flag":[high_flag],
            "Low_Usage_Flag":[low_flag],
            "Prediction":[pred],
            "Theft_Status":[status],
            "AI_Reason":[reason]
})

        old_data = pd.read_csv("electricity_theft_dataset_3000houses.csv")
        updated_data = pd.concat([old_data, new_data], ignore_index=True)
        updated_data.to_csv("electricity_theft_dataset_3000houses.csv", index=False)
        # st.cache_data.clear()
        st.rerun()
        

        st.success(f"House {new_house_id} added successfully!")
        st.markdown("### 🧠 AI Explanation")
        st.info(reason)
        



