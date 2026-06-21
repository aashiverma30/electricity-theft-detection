import streamlit as st
import pandas as pd
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
    df = pd.read_csv("electricity_theft_dataset_3000houses.csv")
except:
    st.error("Dataset not found")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## ⚡ VoltGuard Control")

menu = st.sidebar.radio("", ["📊 Dashboard","🏠 House Monitoring","🚨 Theft Alerts","👨‍💼 Verification"])

month = st.sidebar.selectbox("Select Month", df["Month"].unique())
df = df[df["Month"] == month]

# ---------------- FEATURE ENGINEERING ----------------
df["Difference"] = df["Current_Units"] - df["Average_Units"]
df["Deviation_%"] = ((df["Current_Units"] - df["Average_Units"]) / df["Average_Units"]) * 100
df["Target"] = df["Deviation_%"].apply(lambda x: 1 if x > 50 or x < -35 else 0)

le = LabelEncoder()
df["Month_Encoded"] = le.fit_transform(df["Month"])

# ---------------- MODEL ----------------
X = df[["Current_Units","Average_Units","Month_Encoded","Deviation_%"]]
y = df["Target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier(n_estimators=120)
model.fit(X_train,y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))

df["Prediction"] = model.predict(X)
df["Theft_Status"] = df["Prediction"].apply(lambda x:"Suspicious" if x==1 else "Normal")

# ---------------- DASHBOARD ----------------
if menu == "📊 Dashboard":

    total=len(df)
    suspicious=len(df[df["Theft_Status"]=="Suspicious"])
    normal=total-suspicious

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Houses", total)
    col2.metric("Suspicious ⚡", suspicious)
    col3.metric("Normal", normal)

    if suspicious > total * 0.3:
        st.error("🚨 ALERT: Too many suspicious houses!")

    st.line_chart(df.set_index("House_ID")["Current_Units"])
    st.bar_chart(df.set_index("House_ID")[["Current_Units","Average_Units"]])

    st.success(f"Model Accuracy: {round(accuracy*100,2)}%")

    # ---------------- MANUAL CHECK ----------------
    st.markdown("---")
    st.markdown("## 🔍 Check Electricity Theft")

    current_units = st.number_input("Current Units", 0)
    avg_units = st.number_input("Average Units", 1)

    month_input = st.selectbox("Month Input", df["Month"].unique())

    if st.button("Analyze Usage"):

        deviation = ((current_units - avg_units) / avg_units) * 100
        month_encoded = le.transform([month_input])[0]

        input_df = pd.DataFrame({
            "Current_Units":[current_units],
            "Average_Units":[avg_units],
            "Month_Encoded":[month_encoded],
            "Deviation_%":[deviation]
        })

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error(f"⚠️ Theft Risk HIGH ({round(deviation,2)}%)")
        else:
            st.success(f"✅ Normal Usage ({round(deviation,2)}%)")

        # Reason
        if deviation > 50:
            st.warning("Unusually high usage detected")
        elif deviation < -35:
            st.warning("Sudden drop detected")
        else:
            st.info("Usage looks normal")

    # ---------------- SIMULATOR ----------------
    st.markdown("---")
    st.markdown("## 🎛️ Live Simulator")

    sim_units = st.slider("Adjust Units", 0, 500, 150)
    sim_dev = ((sim_units - 200)/200)*100

    st.write(f"Deviation: {round(sim_dev,2)}%")

    if sim_dev > 50 or sim_dev < -35:
        st.error("⚠️ Suspicious Pattern")
    else:
        st.success("✅ Normal Pattern")

# ---------------- HOUSE ----------------
elif menu == "🏠 House Monitoring":

    house_id = st.selectbox("Select House ID",df["House_ID"].unique())
    house = df[df["House_ID"]==house_id]

    st.dataframe(house)

    if house["Theft_Status"].values[0]=="Suspicious":
        st.error("⚡ Abnormal Usage")
    else:
        st.success("Normal Usage")

# ---------------- ALERTS ----------------
elif menu == "🚨 Theft Alerts":

    alerts = df[df["Theft_Status"]=="Suspicious"]

    if alerts.empty:
        st.success("No Theft Detected")
    else:
        st.dataframe(alerts)

# ---------------- VERIFICATION ----------------
elif menu == "👨‍💼 Verification":

    st.subheader("Inspector Panel")

    suspicious_ids = df[df["Theft_Status"]=="Suspicious"]["House_ID"].unique()

    if len(suspicious_ids)>0:

        house_id = st.selectbox("Select House",suspicious_ids)

        decision = st.radio("Decision",["Confirm Theft","False Alarm","Under Investigation"])
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

            updated.to_csv("verification_log.csv",index=False)
            st.success("Saved")

    else:
        st.info("No suspicious houses")

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