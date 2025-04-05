import os
import pickle
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🧑‍⚕"
)

# Get the working directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved models
diabetes_model_path = os.path.join(working_dir, r"/Users/hardihiteshmakwana/Desktop/Hardi_Makwana_PredictionofDiseaseOutbreaks/diabetes_model.sav")
heart_disease_model_path = os.path.join(working_dir, r"/Users/hardihiteshmakwana/Desktop/Hardi_Makwana_PredictionofDiseaseOutbreaks/heart_model.sav")
parkinsons_model_path = os.path.join(working_dir, r"/Users/hardihiteshmakwana/Desktop/Hardi_Makwana_PredictionofDiseaseOutbreaks/parkinsons_model.sav")

try:
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
    heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
    parkinsons_model = pickle.load(open(parkinsons_model_path, 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading models: {e}")
    st.stop()

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Prediction of Disease Outbreaks System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0
    )

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        SkinThickness = st.text_input("Skin Thickness value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
        Glucose = st.text_input("Glucose Level")
        Insulin = st.text_input("Insulin Level")
        Age = st.text_input("Age of the Person")
    with col3:
        BloodPressure = st.text_input("Blood Pressure value")
        BMI = st.text_input("BMI value")

    if st.button("Diabetes Test Result"):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])[0]

            # Display prediction
            if diab_prediction == 1:
                st.error("The person *has diabetes*.")
            else:
                st.success("The person *does not have diabetes*.")

            # Risk scoring
            risk_score = (
                (float(Glucose) / 200) * 0.4 +  # Glucose levels
                (float(BMI) / 50) * 0.2 +      # BMI
                (float(Age) / 100) * 0.2 +     # Age
                (float(BloodPressure) / 120) * 0.2  # Blood Pressure
            )
            st.subheader(f"Personalized Risk Score for Diabetes: {risk_score:.2f}")

            # 🎯 Add Pie Chart for Diabetes
            diabetes_labels = ['Glucose', 'BMI', 'Age', 'Blood Pressure']
            diabetes_weights = [
                (float(Glucose) / 200) * 0.4,
                (float(BMI) / 50) * 0.2,
                (float(Age) / 100) * 0.2,
                (float(BloodPressure) / 120) * 0.2
            ]
            fig2, ax2 = plt.subplots(facecolor='black')
            ax2.pie(diabetes_weights, labels=diabetes_labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax2.axis('equal')
            fig2.patch.set_facecolor('black')
            st.pyplot(fig2)

            
            if diab_prediction == 1 or risk_score > 1:
                st.warning("⚠️ High Risk: The person is likely to have diabetes. Consult a doctor.")
            else:
                if risk_score > 0.7:
                    st.warning("⚠️ High Risk for developing diabetes. Monitor your health closely.")
                elif risk_score > 0.4:
                    st.info("🟡 Moderate Risk: Maintain a healthy lifestyle.")
                else:
                    st.success("🟢 Low Risk: Keep up the healthy habits!")

        except ValueError:
            st.error("Please provide valid numeric inputs.")

        except ValueError:
            st.error("Please provide valid numeric inputs.")

# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Resting Blood Pressure")
        restecg = st.text_input("Resting Electrocardiographic results")
        oldpeak = st.text_input("ST depression induced by exercise")
        thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect")
    with col2:
        sex = st.text_input("Sex")
        chol = st.text_input("Serum Cholesterol in mg/dl")
        thalach = st.text_input("Maximum Heart Rate achieved")
        slope = st.text_input("Slope of the peak exercise ST segment")
    with col3:
        cp = st.text_input("Chest Pain types")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl")
        exang = st.text_input("Exercise Induced Angina")
        ca = st.text_input("Major vessels colored by fluoroscopy")

    if st.button("Heart Disease Test Result"):
        try:
            user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                          float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            heart_prediction = heart_disease_model.predict([user_input])[0]

            # Display prediction
            if heart_prediction == 1:
                st.error("The person *has heart disease*.")
            else:
                st.success("The person *does not have heart disease*.")

            # Risk scoring
            risk_score = (
                (float(age) / 100) * 0.3 +     # Age
                (float(chol) / 400) * 0.3 +   # Cholesterol levels
                (float(trestbps) / 200) * 0.2 +  # Resting Blood Pressure
                (float(oldpeak) / 6.0) * 0.2  # ST depression
            )
            st.subheader(f"Personalized Risk Score for Heart Disease: {risk_score:.2f}")

            # ✅ PIE CHART for risk factor contribution
            heart_labels = ['Age', 'Cholesterol', 'Resting BP', 'ST Depression']
            heart_weights = [
                (float(age) / 100) * 0.3,
                (float(chol) / 400) * 0.3,
                (float(trestbps) / 200) * 0.2,
                (float(oldpeak) / 6.0) * 0.2
            ]
            fig1, ax1 = plt.subplots(facecolor='black')
            ax1.pie(heart_weights, labels=heart_labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax1.axis('equal')
            fig1.patch.set_facecolor('black')
            st.pyplot(fig1)

            # Risk message based on prediction and score
            if heart_prediction == 1:
                st.warning("*High Risk:* The person has been diagnosed with heart disease. Consult a cardiologist immediately.")
            else:
                if risk_score > 0.7:
                    st.warning("High Risk for developing heart disease. Regular check-ups are recommended.")
                elif risk_score > 0.4:
                    st.info("Moderate Risk for developing heart disease. Maintain a healthy lifestyle.")
                else:
                    st.success("Low Risk for heart disease. Keep up the good work!")

        except ValueError:
            st.error("Please provide valid numeric inputs.")

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    cols = st.columns(5)
    input_labels = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR",
        "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    user_input = []
    for i, label in enumerate(input_labels):
        with cols[i % 5]:
            input_value = st.text_input(label)
            if input_value:
                user_input.append(input_value)
            else:
                user_input.append('0')  # Default value if input is empty

    if st.button("Parkinson's Test Result"):
        try:
            # Ensure all inputs are numeric
            user_input = [float(x) for x in user_input]  # Convert inputs to floats
            parkinsons_prediction = parkinsons_model.predict([user_input])[0]

            # Display prediction
            if parkinsons_prediction == 1:
                st.error("The person *has Parkinson's disease*.")
            else:
                st.success("The person *does not have Parkinson's disease*.")

            # Risk scoring
            parkinsons_risk_score = (
                (user_input[0] / 200) * 0.3 +  # MDVP:Fo(Hz)
                (user_input[3] / 0.1) * 0.3 +  # MDVP:Jitter(%)
                (user_input[15] / 50) * 0.2 +  # HNR
                (user_input[-1] / 1.0) * 0.2   # PPE
            )
            st.subheader(f"Personalized Risk Score for Parkinson's Disease: {parkinsons_risk_score:.2f}")

            # 🎯 Add Pie Chart for Parkinson’s
            parkinsons_labels = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'HNR', 'PPE']
            parkinsons_weights = [
                (user_input[0] / 200) * 0.3,
                (user_input[3] / 0.1) * 0.3,
                (user_input[15] / 50) * 0.2,
                (user_input[-1] / 1.0) * 0.2
            ]
            fig3, ax3 = plt.subplots(facecolor='black')
            ax3.pie(parkinsons_weights, labels=parkinsons_labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax3.axis('equal')
            fig3.patch.set_facecolor('black')
            st.pyplot(fig3)

            if parkinsons_prediction > 1:
                st.warning("*High Risk:* The person has been diagnosed with Parkinson's disease. Consult a neurologist for treatment.")
            else:
                if parkinsons_risk_score > 0.7:
                    st.warning("High Risk for developing Parkinson's disease. Regular check-ups are recommended.")
                elif parkinsons_risk_score > 0.4:
                    st.info("Moderate Risk for developing Parkinson's disease. Maintain a healthy lifestyle.")
                else:
                    st.success("Low Risk for Parkinson's disease. Keep up the good work!")

        except ValueError:
            st.error("Please provide valid numeric inputs for all fields.")
