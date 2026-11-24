"""VoltBot - rule-based chatbot for VoltGuard AI (English only).

Works fully offline: it answers from the dataframe that app.py already builds
(Risk_Score, Risk_Level, Theft_Status, AI_Reason, Deviation_%, Usage_Ratio ...).
"""
import os
import re

import pandas as pd
import streamlit as st

LOSS_PER_CASE = 5000  # same hypothetical value used in the Dashboard panel

SUGGESTIONS = [
    "How many houses are suspicious?",
    "Top 5 risky houses",
    "What is the revenue loss?",
    "Verification summary",
]

HELP_TEXT = (
    "Here is what I can help you with:\n\n"
    "- **Summary**: `how many houses are suspicious?`, `total houses`, `how many are normal?`\n"
    "- **Top risky houses**: `top 10 risky houses`, `highest risk houses`\n"
    "- **House details**: `status of house 25`, `why is house 120 flagged?`\n"
    "- **By month**: `how many suspicious in March?`, `top 5 risky houses in June`\n"
    "- **Impact**: `revenue loss`, `grid impact`\n"
    "- **Inspector log**: `verification summary`, `history of house 25`\n"
    "- **About the model**: `how does the model work?`, `what is a risk score?`\n"
)


# ---------------- helpers ----------------
def _has(text, words):
    return any(w in text for w in words)


def _find_month(text, df, default):
    for m in df["Month"].unique():
        if re.search(rf"\b{re.escape(str(m).lower())}\b", text):
            return m
    return default


def _find_house_id(text):
    m = re.search(r"(?:house|home|id)\s*(?:no\.?|number|#)?\s*(\d+)", text)
    if m:
        return int(m.group(1))
    if "top" not in text:
        nums = re.findall(r"\b\d+\b", text)
        if nums:
            return int(nums[0])
    return None


def _find_top_n(text, default=5):
    m = re.search(r"top\s*(\d+)", text) or re.search(r"(\d+)\s*(?:risky|highest|most)", text)
    return max(1, min(int(m.group(1)), 50)) if m else default


def _load_log():
    if not os.path.exists("verification_log.csv"):
        return pd.DataFrame()
    try:
        return pd.read_csv("verification_log.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame()


def _prepare(df):
    """Make sure every column the bot needs exists, deriving missing ones."""
    df = df.copy()
    cur = pd.to_numeric(df["Current_Units"], errors="coerce")
    avg = pd.to_numeric(df["Average_Units"], errors="coerce").replace(0, 1)

    if "Deviation_%" not in df.columns:
        df["Deviation_%"] = (cur - avg) / avg * 100
    if "Usage_Ratio" not in df.columns:
        df["Usage_Ratio"] = cur / avg

    # severity: 1.0 means exactly on the project's flagging thresholds
    sev = pd.concat(
        [df["Deviation_%"] / 45, -df["Deviation_%"] / 30, (df["Usage_Ratio"] - 1) / 0.6], axis=1
    ).max(axis=1).clip(lower=0, upper=1.33)

    if "Risk_Score" not in df.columns:
        df["Risk_Score"] = (sev * 75).clip(upper=100).round(1)
    if "Risk_Level" not in df.columns:
        df["Risk_Level"] = pd.cut(
            df["Risk_Score"], [-1, 39.999, 74.999, 101], labels=["Low", "Medium", "High"]
        ).astype(str)
    if "Theft_Status" not in df.columns:
        df["Theft_Status"] = (sev >= 1).map({True: "Suspicious", False: "Normal"})
    if "AI_Reason" not in df.columns:
        def _reason(row):
            r = []
            if row["Deviation_%"] > 50:
                r.append("Unusually high consumption")
            if row["Deviation_%"] < -35:
                r.append("Sudden drop in usage")
            if row["Usage_Ratio"] > 1.5:
                r.append("Usage ratio too high")
            if row["Current_Units"] > 400:
                r.append("Very high usage detected")
            if row["Current_Units"] < 100:
                r.append("Very low usage (possible tampering)")
            return ", ".join(r) if r else "Normal usage"
        df["AI_Reason"] = df.apply(_reason, axis=1)
    return df


# ---------------- answer engine ----------------
def answer(query, df, selected_month):
    """Return (markdown_text, optional_dataframe)."""
    df = _prepare(df)
    text = query.lower().strip()
    month = _find_month(text, df, selected_month)
    mdf = df[df["Month"] == month]
    suspicious = mdf[mdf["Theft_Status"] == "Suspicious"]
    hid = _find_house_id(text)

    # greeting / help
    if text.strip("!. ") in {"hi", "hello", "hey", "hii", "good morning", "good evening"} or _has(
        text, ["help", "what can you do", "what can you help"]
    ):
        return "Hello! 👋 I'm **VoltBot**, the VoltGuard assistant.\n\n" + HELP_TEXT, None

    # verification / inspector log
    if _has(text, ["verification", "inspector", "decision", "history", "log"]):
        log = _load_log()
        if log.empty or "Decision" not in log.columns:
            return "There are no verification records yet.", None
        if hid is not None:
            h = log[log["House_ID"] == hid]
            if h.empty:
                return f"No verification records found for house **{hid}**.", None
            return f"Verification record(s) for house **{hid}** ({len(h)} found):", h
        c = log["Decision"].value_counts()
        return (
            f"**Verification summary** ({len(log)} records in total)\n\n"
            f"- Confirmed Theft: **{int(c.get('Confirm Theft', 0))}**\n"
            f"- False Alarm: **{int(c.get('False Alarm', 0))}**\n"
            f"- Under Investigation: **{int(c.get('Under Investigation', 0))}**"
        ), log.tail(5).iloc[::-1]

    # top / highest risk
    if _has(text, ["top", "highest", "most risky", "risky", "worst"]) and not re.search(r"(house|home)\s*\d+", text):
        n = _find_top_n(text)
        top = mdf.sort_values("Risk_Score", ascending=False).head(n)
        cols = ["House_ID", "Current_Units", "Average_Units", "Deviation_%", "Risk_Score", "Risk_Level", "AI_Reason"]
        return f"Top {len(top)} riskiest houses in **{month}**:", top[cols]

    # single house lookup
    if hid is not None and (
        _has(text, ["house", "home", "id", "status", "risk", "why", "flag", "check", "detail", "show", "tell"])
        or len(text.split()) <= 3
    ):
        h = df[df["House_ID"] == hid]
        if h.empty:
            return f"House **{hid}** was not found in the dataset.", None
        row = h[h["Month"] == month]
        row = row.iloc[0] if not row.empty else h.iloc[-1]
        verdict = "🚨 Suspicious" if row["Theft_Status"] == "Suspicious" else "✅ Normal"
        return (
            f"**House {hid}** ({row['Month']})\n\n"
            f"- Status: **{verdict}**\n"
            f"- Risk: **{row['Risk_Level']}** ({row['Risk_Score']:.0f}%)\n"
            f"- Current / Average units: **{row['Current_Units']:.0f}** / **{row['Average_Units']:.0f}**\n"
            f"- Deviation: **{row['Deviation_%']:.1f}%**, Usage ratio: **{row['Usage_Ratio']:.2f}**\n"
            f"- AI reason: {row['AI_Reason']}\n\n"
            "_Note: abnormal usage does not confirm theft. An inspection is required._"
        ), None

    # revenue / impact
    if _has(text, ["revenue", "loss", "impact", "grid", "discom"]):
        n = len(suspicious)
        return (
            f"Impact estimate for **{month}**:\n\n"
            f"- Suspicious houses: **{n}**\n"
            f"- Estimated revenue risk: **₹ {n * LOSS_PER_CASE:,}** (₹{LOSS_PER_CASE:,} per case, hypothetical)\n"
            f"- Zones needing inspection: **{n}**"
        ), None

    # counts / summary
    if _has(text, ["suspicious", "theft", "alert", "how many", "total", "normal", "summary", "overview"]):
        total, n = len(mdf), len(suspicious)
        high = int((suspicious["Risk_Level"] == "High").sum())
        pct = (n / total * 100) if total else 0
        return (
            f"Summary for **{month}**:\n\n"
            f"- Total houses: **{total}**\n"
            f"- Suspicious: **{n}** ({pct:.1f}%), of which High risk: **{high}**\n"
            f"- Normal: **{total - n}**"
        ), None

    # model / explanation
    if _has(text, ["model", "how does", "how do", "algorithm", "random forest", "score", "feature", "accuracy"]):
        return (
            "**How the VoltGuard model works**\n\n"
            "- A **Random Forest** classifier labels every house-month as Normal or Suspicious.\n"
            "- Features: current and average units, deviation %, usage ratio, high/low usage flags and month.\n"
            "- The **risk score** is the model's probability (0-100%) that a house is suspicious.\n"
            "- Risk levels: **High** ≥ 75, **Medium** 40-74, **Low** < 40.\n"
            "- A flag only means an inspection is needed. The final decision is made by an inspector."
        ), None

    return (
        "Sorry, I didn't understand that. 😅 Type `help` to see what I can answer.",
        None,
    )


# ---------------- UI ----------------
def render_chatbot(df, month):
    st.subheader("🤖 VoltBot - Ask about your data")
    st.caption(f"Default month: **{month}** (change it in the sidebar, or mention a month name in your question)")

    if "voltbot_msgs" not in st.session_state:
        st.session_state.voltbot_msgs = [
            {
                "role": "assistant",
                "content": "Hello! 👋 I'm **VoltBot**. Try one of the buttons below or type your question.",
                "table": None,
            }
        ]

    cols = st.columns(len(SUGGESTIONS))
    pending = None
    for col, s in zip(cols, SUGGESTIONS):
        if col.button(s, key=f"voltbot_sugg_{s}", use_container_width=True):
            pending = s

    for msg in st.session_state.voltbot_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)

    prompt = st.chat_input("Ask a question... (e.g. status of house 25)") or pending
    if prompt:
        reply, table = answer(prompt, df, month)
        st.session_state.voltbot_msgs.append({"role": "user", "content": prompt, "table": None})
        st.session_state.voltbot_msgs.append({"role": "assistant", "content": reply, "table": table})
        st.rerun()

    if len(st.session_state.voltbot_msgs) > 1 and st.button("🧹 Clear chat", key="voltbot_clear"):
        del st.session_state["voltbot_msgs"]
        st.rerun()
