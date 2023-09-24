from flask import Flask, render_template, Response, request, redirect, url_for,send_file
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from collections import Counter
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from collections import defaultdict
import csv
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://app_user:app_password@localhost/app_database'  # replace 'password' with the actual password
db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'  # Specify the table name since it's not the default plural form

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    phone_number = db.Column(db.String, unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Result(db.Model):
    __tablename__ = 'results'  # Specify the table name since it's not the default plural form

    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    emotion_data = db.Column(db.JSON)

# Initialize global variables
emotion_predictions = []
data = []
model_running = False
start_predict = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/carousel')
def carousel():
    global model_running
    model_running = True
    return render_template('carousel.html')


@app.route('/stop_model', methods=['GET'])
def stop_model():
    try:
        emotion_predictions = []
        global model_running
        model_running = False
        return redirect(url_for('predictions'))
    except Exception as e:
        return str(e)

@app.route('/predictions')
def predictions():
    global emotion_predictions
    global data
    max_percentages = []
    default_dict = {emotion: 0.0 for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
    for image_predictions in emotion_predictions:
        total = sum(image_predictions.values())
        print(f"total: {total}")  # debug print
        print(f"image_predictions: {image_predictions}")  # debug print
        percentages = {emotion: round(count / total * 100, 2) for emotion, count in image_predictions.items()}
        updated_dict = default_dict.copy()
        updated_dict.update(percentages)
        print(f"updated_dict: {updated_dict}")  # debug print
        max_percentages.append(max(updated_dict.values()))  
        data.append(updated_dict)
    emotion_predictions = []
    print(f"data: {data}")  # debug print
    return render_template('predictions.html', percentage_predictions=data, max_percentages=max_percentages)

def max_emotion(image_predictions):
    return max(image_predictions, key=image_predictions.get)

# Add a new user
@app.route('/add_user/<phone>')
def add_user(phone):
    new_user = User(phone_number=phone)
    db.session.add(new_user)
    db.session.commit()
    return f"User {phone} added successfully!"

# Query all users
@app.route('/list_users')
def list_users():
    users = User.query.all()
    return '\n'.join([user.phone_number for user in users])

# Set up Jinja environment and custom filters
env = Environment(
    loader=FileSystemLoader('templates'),  # change this line
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True,
)
env.filters['max_emotion'] = max_emotion

# Load the model
json_file = open('new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('new_model.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    global model_running
    global emotion_predictions 
    global start_predict
    cap = cv2.VideoCapture(0)
    image_number = 0
    start_time = datetime.now()
    
    last_image = 17
    
    while model_running:
        
        ret, frame = cap.read()

        if not ret:
            break
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_image)
        
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray_image[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))

            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)

            if start_predict == True:
                model_output = model.predict(image_pixels)
                max_index = np.argmax(model_output[0])

                emotion_detection = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                emotion_prediction = emotion_detection[max_index]

                cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"emotion_prediction = {emotion_prediction}")
                if model_running:
                    # Check if 3 seconds have passed
                    if (datetime.now() - start_time).seconds >= 3:
                        # Move to next image
                        image_number += 1
                        start_time = datetime.now()

                    # Append the prediction to the correct image dictionary
                    if len(emotion_predictions) < image_number:
                        emotion_predictions.append({})

                    emotion_predictions[image_number - 1][emotion_prediction] = emotion_predictions[image_number - 1].get(emotion_prediction, 0) + 1
                    print(f"emotion_predictions: {emotion_predictions}")  # debug print
  
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        if image_number == last_image:
            break
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/button_clicked', methods=['POST'])
def button_clicked():
    global start_predict
    start_predict = True
    return ''

@app.route('/download_report', methods=['POST'])
def download_report():
    global emotion_predictions
    global data
    
    print(data)

    # Write data to CSV file
    filename = 'report.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Image', 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        writer.writeheader()
        for i, row in enumerate(data, start=1):
            writer.writerow({'Image': f'Image {i}', **row})

    emotion_predictions = []  # Clear emotion predictions after CSV file is written

    # Send file for download
    response = send_file(filename, mimetype='text/csv', as_attachment=True)
    response.headers["Content-Disposition"] = "attachment; filename=report.csv"
    return response


if __name__ == '__main__':
    app.run(port=5000)