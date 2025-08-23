from flask import Blueprint
from datetime import datetime
import random

main = Blueprint('main', __name__)

@main.route('/about')
def about():
    return "About Page"

@main.route('/detect')
def detect():
    detected = random.choice([True, False])
    velocity = random.randint(800, 1500)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if detected:
        result = f"CME Detected \nVelocity: {velocity} km/s\nTime: {timestamp}"
    else:
        result = f"No CME Activity \nTime: {timestamp}"

    return result
