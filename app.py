import streamlit as st
import pandas as pd
import pickle


""", unsafe_allow_html=True)

# ------------------- LOAD MODEL -------------------
file_to_load = "xgboost_reunion_model.pkl"   # Must be in SAME folder
with open(file_to_load, "rb") as file:
    loaded_model = pickle.load(file)

# ------------------- UI LAYOUT -------------------
st.set_page_config(page_title="🎓 Reunion Attendance Predictor", layout="centered")
st.title("🎓 Alumni Reunion Attendance Predictor")
st.markdown("Enter alumni information below to estimate the probability of attending.")

st.markdown("---")

# ------------------- USER INPUTS -------------------
bucknell_year = st.slider("Bucknell Class Year", 1950, 2025, 2000)

peer_to_peer = st.checkbox("Peer-to-Peer Invite?")
prospect = st.checkbox("Prospect")
greek_sorority = st.checkbox("Greek Life – Sorority")
volunteer = st.checkbox("Volunteer")
donor = st.checkbox("Donor")

cornerstone_potential = st.checkbox("Cornerstone Status: Potential Renew")
cornerstone_renew = st.checkbox("Cornerstone Status: Renew")

event_info = st.checkbox("Received Event Information?")

num_volunteer_activities = st.slider(
    "Number of Volunteer Activities",
    min_value=0,
    max_value=50,
    value=0
)

st.markdown("---")

# ------------------- PREDICTION -------------------
if st.button("Predict Attendance"):

    new_alumni = pd.DataFrame({
        'Bucknell_Class_Year': [bucknell_year],
        'Peer-to-Peer': [int(peer_to_peer)],
        'Prospect': [int(prospect)],
        'Greek_Type_Sorority': [int(greek_sorority)],
        'Cornerstone_Status_POTENTIAL RENEW CS': [int(cornerstone_potential)],
        'Volunteer': [int(volunteer)],
        'Event_Information': [int(event_info)],
        'Cornerstone_Status_RENEW CS': [int(cornerstone_renew)],
        'Number_Volunteer_Activities': [num_volunteer_activities],
        'Donor': [int(donor)]
    })

    predicted_prob = loaded_model.predict_proba(new_alumni)[:, 1]
    predicted_class = loaded_model.predict(new_alumni)

    formatted_prob = f"{predicted_prob[0]:.2f}"

    st.markdown("## ✅ Prediction Results")
    st.write(f"**Predicted Probability of Attending:** {formatted_prob}")

    if predicted_class[0] == 1:
        st.success("🎉 Predicted Outcome: **Will Attend**")
    else:
        st.error("❌ Predicted Outcome: **Will Not Attend**")

    # ------------------- VISUAL BREAKDOWN -------------------
    probabilities = [1 - predicted_prob[0], predicted_prob[0]]
    labels = ['Will Not Attend', 'Will Attend']
    chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)

    st.markdown("### 📊 Prediction Breakdown")
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("**Bucknell Reunion Attendance ML App** | Powered by Streamlit")
