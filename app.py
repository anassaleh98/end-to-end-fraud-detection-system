from flask import Flask, request, render_template, redirect, url_for, flash, send_file, send_from_directory
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from pipeline import predict_pipeline
import uuid

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model pipeline (adjust path for portability)
model_pipeline = joblib.load(r"E:\My_Github\Credit Card Fraud Detection\models\stacking_pipeline.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = {}
        features['Time'] = float(request.form.get('Time', 0))
        features['Amount'] = float(request.form.get('Amount', 0))

        for i in range(1, 29):
            val = request.form.get(f'V{i}', "0")
            features[f'V{i}'] = float(val)

        input_df = pd.DataFrame([features])
        preds, probs = predict_pipeline(model_pipeline, input_df)

        result = "FRAUD" if preds[0] == 1 else "LEGITIMATE"
        confidence = round(probs[0] * 100, 2)

        return render_template('index.html', result=result, confidence=confidence)

    except Exception as e:
        app.logger.exception(e)
        flash(f"Error processing input: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            flash("No file uploaded!", "danger")
            return redirect(url_for('home'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        if not all(col in df.columns for col in required_cols):
            flash("CSV missing required columns!", "danger")
            return redirect(url_for('home'))

        preds, probs = predict_pipeline(model_pipeline, df)
        df['Prediction'] = preds
        df['Fraud Probability'] = probs

        results_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "batch_results.csv")
        df.to_csv(results_csv_path, index=False)

        # Save chart with unique filename
        chart_filename = f"chart_{uuid.uuid4().hex}.png"
        chart_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_filename)

        # ✅ Fraud vs Legit Count bar chart
        plt.figure(figsize=(6, 4))
        counts = df['Prediction'].value_counts()
        plt.bar(['Legitimate', 'Fraud'], counts, color=['green', 'red'])
        plt.title("Fraud vs Legitimate Transactions")
        plt.ylabel("Number of Transactions")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        # Save dataset for pagination use
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], "paginated_results.csv")
        df.to_csv(session_file, index=False)

        return redirect(url_for("batch_results", page=1, fraud_page=1, chart_filename=chart_filename))

    except Exception as e:
        app.logger.exception(e)
        flash(f"Error processing CSV: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/batch_results')
def batch_results():
    try:
        page = int(request.args.get("page", 1))              # all transactions page
        fraud_page = int(request.args.get("fraud_page", 1))  # fraud transactions page
        chart_filename = request.args.get("chart_filename", None)
        per_page = 10

        session_file = os.path.join(app.config['UPLOAD_FOLDER'], "paginated_results.csv")
        df_display = pd.read_csv(session_file)

        # Fraud-only rows
        fraud_transactions = df_display[df_display['Prediction'] == 1]

        # Paginate all transactions
        start = (page - 1) * per_page
        end = start + per_page
        df_page = df_display.iloc[start:end]

        # Paginate fraud transactions
        f_start = (fraud_page - 1) * per_page
        f_end = f_start + per_page
        fraud_page_df = fraud_transactions.iloc[f_start:f_end]

        def highlight_fraud(row):
            return ['background-color: #f8d7da; color: #721c24;' if row['Prediction'] == 1 else '' for _ in row]

        # Apply highlighting only to the general transactions table
        styled_table = df_page.style.apply(highlight_fraud, axis=1)
        
        # Create fraud table WITHOUT highlighting (just normal pandas table)
        fraud_styled_table = fraud_page_df.style

        total_pages = int(np.ceil(len(df_display) / per_page))
        fraud_total_pages = int(np.ceil(len(fraud_transactions) / per_page)) if len(fraud_transactions) > 0 else 1

        return render_template(
            'batch_result.html',
            table_html=styled_table.to_html(),
            fraud_table_html=fraud_styled_table.to_html(),
            chart=url_for('uploaded_file', filename=chart_filename) if chart_filename else None,
            download_link=url_for('download_results'),
            current_page=page,
            total_pages=total_pages,
            fraud_page=fraud_page,
            fraud_total_pages=fraud_total_pages
        )
    except Exception as e:
        app.logger.exception(e)
        flash(f"Error displaying batch results: {str(e)}", "danger")
        return redirect(url_for('home'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/download_results')
def download_results():
    results_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], "batch_results.csv")
    return send_file(results_csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)