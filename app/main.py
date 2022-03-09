from flask import Flask, render_template, request, jsonify
from app.prediction import PersonalityPrediction

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    result = "Not Found"
    if request.method == 'POST':
        person = PersonalityPrediction()
        result = person.predict(request.get_json())
        return jsonify(result)