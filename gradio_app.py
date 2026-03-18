import pandas as pd
import numpy as np
import xgboost as xgb
import gradio as gr
import pickle

# Load model, scaler
xgb_model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

# Mapping for loan_intent (same as training)
loan_intent_mapping = {
    'EDUCATION': 'EPM',
    'PERSONAL': 'EPM', 
    'MEDICAL': 'EPM',
    'DEBTCONSOLIDATION': 'EPM',
    'VENTURE': 'VENTURE',
    'HOMEIMPROVEMENT': 'HOMEIMPROVEMENT'
}

def predict_loan_status(person_income, person_home_ownership, 
                        loan_intent, previous_loan_defaults_on_file, 
                        loan_int_rate, loan_amount):
    # for person with income less than 10000; if loan_amount <= person_income/10: then approve the loan, else use the model to predict
    if loan_amount < 10:
        return "✅ APPROVED"
    elif person_income < 10000 and loan_amount <= person_income / 10:
        return "✅ APPROVED"
    elif person_income > 1000000 and loan_amount <= person_income / 50:
        return "✅ APPROVED"
    elif person_income > 10000 and person_income <= 1000000 and loan_amount <= 1000:
        return "✅ APPROVED"
    elif person_income <= loan_amount /2:
        return "❌ REJECTED"
    # Apply loan_intent mapping (defensive check)
    if loan_intent in loan_intent_mapping:
        loan_intent_mapped = loan_intent_mapping[loan_intent]
    else:
        loan_intent_mapped = 'EPM'  # default fallback

    input_data = {
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent_mapped,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_amount / person_income if person_income > 0 else 100
    }

    column_order = ['person_income', 'person_home_ownership', 'loan_intent',
                    'loan_int_rate', 'loan_percent_income','previous_loan_defaults_on_file']

    input_df = pd.DataFrame([input_data])[column_order]

    # 3. Encode using the saved encoder

    categorical_cols = ["person_home_ownership","loan_intent","previous_loan_defaults_on_file"]


    data_encoded = encoder.transform(input_df[categorical_cols])
    input_df = input_df.drop(categorical_cols, axis=1)
    input_df = pd.concat([input_df, pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(), index=input_df.index)], axis=1)
    # 4. Scale using the saved scaler
    input_scaled = scaler.transform(input_df)

    # 5. Prediction Logic
    prediction = xgb_model.predict(input_scaled)[0]


    # Your new code:
    return "✅ APPROVED" if prediction == 1 else "❌ REJECTED"

# Define inputs - MUST match function parameter order
inputs = [
    gr.Number(label="Annual Income ($)"),
    gr.Dropdown([ "OWN", "MORTGAGE", "RENT","OTHER" ],label="Home Ownership"),
    gr.Dropdown(label="Loan Intent", choices=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]),
    gr.Dropdown(["Yes", "No"], label="Previous Loan Defaults"),
    gr.Number(label="Interest Rate (%)"),
    gr.Number(label="Loan Amount ($)")
]

# Define output
output = gr.Textbox(label="Prediction Result")

# Launch interface
gr.Interface(fn=predict_loan_status, inputs=inputs, outputs=output, title="Loan Approval Prediction").launch(share=True)
