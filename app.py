
from model import encode_data, load_model, predict_model
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd


app = Flask(__name__, template_folder="templates")
import modelbit
mb = modelbit.login()
# Load the model on startup
#
# Load existing data (or create an empty DataFrame with columns)
file_path = ""

try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    data = pd.DataFrame(columns=[])

@app.route("/")
def home():
    return render_template("upload_form.html")

@app.route("/predict",methods=["POST"])
def predict():
    # model = train_model()
    # Check if the file is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        input_df = pd.read_csv(file)
    # Preprocess the input data
    original_df = input_df.copy()
    input_df.to_csv('original_upload_csv.csv')
    encoded_df = encode_data(input_df)
    print(encoded_df.columns)
    if 'PerformanceRating' in encoded_df.columns:
        X = encoded_df.drop(columns=["PerformanceRating"],axis=1)
        original_df=original_df.drop(columns=["PerformanceRating"],axis=1)
    else:
        return X
    # Make a prediction
    print(f'the X columns {X.columns}')
    model = load_model()
    prediction = predict_model(model, X)
    print(prediction)
    # data.to_csv('pred_accident_data_to_predict.csv', index=False)
    # Return the result
    
    X["Predicted PerformaceRating"] = prediction

    original_df["Predicted PerformanceRating"] = prediction

    # Save the updated DataFrame to a new CSV
    # X.to_csv('output_cleaned_csv.csv', index=False)
    # Save the updated DataFrame to a new CSV
    X.to_csv("outpu_original_csv.csv", index=False)
    # data = pd.read_csv("output_original_csv.csv")
    
    table = (
        original_df.to_html(classes="table table-striped")
    )
    return render_template("result.html", prediction=table[:-1])
    # return jsonify({'prediction': prediction})


## run the app
if __name__ == "__main__":
    app.run(debug=True)
