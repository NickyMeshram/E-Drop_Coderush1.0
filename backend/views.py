from flask import Blueprint, render_template, jsonify
import random
import time

views = Blueprint('views', __name__)

@views.route('/')
def home():
    return render_template('dashboard.html')

@views.route('/dashboard')
def dashboard():
    # Sample data to display
    items = ['Item 1', 'Item 2', 'Item 3']
    return render_template('dashboard.html', items=items)

@views.route('/detect')
def detect_cme():
    cme_detected = random.random() > 0.7
    result = {
        "status": "CME Detected 🚨" if cme_detected else "No CME Activity",
        "velocity": "1200 km/s" if cme_detected else "N/A",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(result)
