from flask import Flask, render_template, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import os
from jinja2 import FileSystemLoader
from flask import send_from_directory
import csv
from datetime import datetime
from werkzeug.utils import secure_filename
from predict import get_prediction
from hyperopt_generate_pipeline import pipeline_generator
from tpot_generate_pipeline import tpot_pipeline_generator
import html
import json



app = Flask(__name__, static_url_path='')
# Configure Flask to use the static folder for templates
template_loader = FileSystemLoader(searchpath=os.path.join(app.root_path, 'static'))
app.jinja_loader = template_loader


# Configure logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("server.log", maxBytes=10000, backupCount=1)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

# Define global variables
prediction_result = ""
fullPath = ""


UPLOAD_FOLDER = './uploads/'
RESULTS_FOLDER = './results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def ensure_uploads_directory():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

def ensure_results_directory():
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        csv_file = os.path.join(RESULTS_FOLDER, 'results.csv')
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'filePath', 'textData', 'usedModel', 'prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

def save_to_csv(data):
    csv_file = os.path.join(RESULTS_FOLDER, 'results.csv')
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'filePath', 'textData', 'usedModel', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data)


@app.route('/')
def get_index():
    response = app.send_static_file('workflow.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/workflow')
def newpage():
    app.logger.info(f"Prediction: {prediction_result}, Full path: {fullPath}")
    return render_template('workflow.html', prediction=prediction_result)


# Define the directory where your images are stored
IMAGE_DIRECTORY = './static/hyperopt-results/'
STATIC_DIRECTORY = './static/'
TPOT_IMAGE_DIRECTORY = './static/tpot-results/'


@app.route('/static/images/<path:filename>')
def serve_loader(filename):
    return send_from_directory(os.path.join(STATIC_DIRECTORY, 'images'), filename)


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(IMAGE_DIRECTORY, 'images'), filename)

@app.route('/dataflows/<path:filename>')
def serve_dataflow(filename):
    return send_from_directory(os.path.join(IMAGE_DIRECTORY, 'dataflows'), filename)

@app.route('/tpot/images/<path:filename>')
def serve_tpot_image(filename):
    return send_from_directory(os.path.join(TPOT_IMAGE_DIRECTORY, 'images'), filename)

@app.route('/tpot/dataflows/<path:filename>')
def serve_tpot_dataflow(filename):
    return send_from_directory(os.path.join(TPOT_IMAGE_DIRECTORY, 'dataflows'), filename)


@app.route('/api/workflow', methods=['POST'])
def post_workflow():
    global prediction_result, fullPath

    try:
        textData = request.form['textData']
        selectedModel = request.form['selectedModel']
        fileName = request.form['fileName']
        
        file = request.files['fileUploaded']
        prediction_result = get_prediction(textData, selectedModel)
        ensure_uploads_directory()

        fullPath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fileName))
        file.save(fullPath)

        data = {
            "data": {
                "filePath": fullPath,
                "textData": textData,
                "usedModel": selectedModel,
                "prediction": prediction_result
            }
        }
       

        ensure_results_directory()
        save_to_csv({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filePath': fullPath,
            'textData': textData,
            'usedModel': selectedModel,
            'prediction': prediction_result
        })

        prediction_str = f"""
        <div style='display: flex; align-items: center; justify-content: space-between; text-align: left; border-radius: 10px; border: 1px solid; width: 35%; padding: 10px;'>
            <p style="margin: 0;"><strong>Predicted Analytical Intent: &nbsp</strong>{data['data']['prediction']}</p>
            <button style='border-radius: 10px; background-color: blue; color: white; padding: 10px 20px;' @click="onNextClick">Next</button>
        </div>
        """

        return prediction_str, 200
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/workflow/tpot', methods=['POST'])
def run_tpot_workflow():
    try:
        img_filename, graph_filename, metric_name, metric_value = tpot_pipeline_generator(fullPath, prediction_result)
        metric_value = round(metric_value, ndigits =2)

        # Return filenames to HTML
        return jsonify({'img_filename': img_filename, 'graph_filename': graph_filename, 'metric_name': metric_name, 'metric_value': metric_value}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflow/hyperopt', methods=['POST'])
def run_hyperopt_workflow():
    try:
        img_filename, graph_filename, metric_name, metric_value = pipeline_generator(fullPath, prediction_result)
        metric_value = round(metric_value, ndigits =2)

        # Return filenames to HTML
        return jsonify({'img_filename': img_filename, 'graph_filename': graph_filename, 'metric_name': metric_name, 'metric_value': metric_value}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.environ.get("FLASK_SERVER_PORT", 9000), debug=True)
