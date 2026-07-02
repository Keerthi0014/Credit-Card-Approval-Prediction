from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["AGE"])
    employment_years = int(request.form["EMPLOYMENT_YEARS"])

    mobile = request.form["MOBILE_NUMBER"]
    email = request.form["EMAIL"]

    data = {
        "CODE_GENDER": request.form["CODE_GENDER"],
"FLAG_OWN_CAR": request.form["FLAG_OWN_CAR"],
"FLAG_OWN_REALTY": request.form["FLAG_OWN_REALTY"],
        "CNT_CHILDREN": int(request.form["CNT_CHILDREN"]),
        "AMT_INCOME_TOTAL": float(request.form["AMT_INCOME_TOTAL"]),
        "NAME_INCOME_TYPE": request.form["NAME_INCOME_TYPE"],
        "NAME_EDUCATION_TYPE": request.form["NAME_EDUCATION_TYPE"],
        "NAME_FAMILY_STATUS": request.form["NAME_FAMILY_STATUS"],
        "NAME_HOUSING_TYPE": request.form["NAME_HOUSING_TYPE"],
        "DAYS_BIRTH": -(age * 365),
        "DAYS_EMPLOYED": -(employment_years * 365),
        "FLAG_MOBIL": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": 1 if mobile else 0,
        "FLAG_EMAIL": 1 if email else 0,
        "OCCUPATION_TYPE": request.form["OCCUPATION_TYPE"],
        "CNT_FAM_MEMBERS": float(request.form["CNT_FAM_MEMBERS"])
    }

    df = pd.DataFrame([data])

    categorical_cols = [
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE"
    ]

    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    prediction = model.predict(df)

    if prediction[0] == 1:
        result = "✅ CREDIT CARD APPROVED"
        message = "Congratulations! The applicant satisfies the approval criteria."
        result_class = "approved"
    else:
        result = "❌ CREDIT CARD REJECTED"
        message = "The applicant does not satisfy the approval criteria."
        result_class = "rejected"

    return render_template(
        "index.html",
        prediction=result,
        message=message,
        result_class=result_class
    )


if __name__ == "__main__":
    app.run(debug=True)