from flask import Blueprint, render_template, jsonify, request, Response
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import os
import yolov5
from threading import Thread
import torch
import time
from pygame import mixer
import base64
from database import db, Fire_Alerts, Fire_Location
from App import create_app
from itertools import zip_longest


model1 = yolov5.load("Models/yolocff.pt")
classes = model1.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'

global rec_frame, switch, neg, rec, out
neg = 0
switch = 0
rec = 0
rec_frame = None
camera = None

# Make shots directory to save pics
try:
    os.mkdir('static/shots')
except OSError as error:
    pass

fire_detected = False  # Flag variable to track if fire is detected

def record(out):
    global rec_frame
    while rec:
        time.sleep(0.05)
        out.write(rec_frame)


def score_frame(frame):
    model1.to(device)
    frame = [frame]
    results = model1(frame)
    print(results)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def class_to_label(x):
    return classes[int(x)]


def plot_boxes(results, frame):
    global fire_detected  # Access the flag variable

    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 4)
            mixer.init()
            sound = mixer.Sound('fire_alarm.ogg')
            sound.play()
            print("fire is detected")
        
        if not fire_detected:
            now = datetime.now()
            p = os.path.join('static', 'shots', 'shot_{}.png'.format(str(now).replace(":", '')))
            cv2.imwrite(p, frame)
            fire_detected = True

            # Read the saved image file as bytes
            with open(p, 'rb') as file:
                image_data = file.read()

            # Encode the image bytes as base64
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            new_alert = Fire_Alerts(date=str(now.date()), time=str(now.time()), image_path=encoded_image)
            db.session.add(new_alert)
            db.session.commit()   
        break
    return frame


def gen_frames(app):  # generate frame by frame from camera
 with app.app_context():
    global rec_frame, camera
    while True:
      if camera is not None:
        success, frame = camera.read()
        if success:
            if switch:
                results = score_frame(frame)
                frame = plot_boxes(results, frame)
            if neg:
                frame = cv2.bitwise_not(frame)
            if rec:
                global rec_frame
                rec_frame = frame
                frame = cv2.putText(frame, "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


def predict_label(img_path):
    # Read image
    model = load_model("Models/fire_smoke_and_nonfire_detection.h5", compile=False)

    # Preprocess image
    test_image = image.load_img(img_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    Catagories = ['Smoke', 'Fire', 'No Fire']

    return Catagories[np.argmax(result)]


View = Blueprint(__name__, "View")


@View.route('/')
def Home():
    return render_template('index.html')


@View.route('/About')
def About():
    return render_template('About.html')


@View.route('/FireAlerts')
def FireAlerts():
    app = create_app()  # Create the Flask app object
    with app.app_context():
    # Retrieve all alerts from the database
      alerts = Fire_Alerts.query.all()
      locations = Fire_Location.query.all()
      combined_data = list(zip_longest(alerts, locations))
    # Render the alert.html template and pass the alerts data
    return render_template('FireAlerts.html', combined_data=combined_data)

@View.route('/data')
def get_data():
    # Calculate the date range for the last 10 days
    today = datetime.now().date()
    ten_days_ago = today - timedelta(days=9)
    
    # Retrieve the data for the last 10 days from the database
    fire_alerts = Fire_Alerts.query.filter(Fire_Alerts.date >= ten_days_ago).all()

    # Calculate the count of fire alerts per day
    data = []
    for i in range(10):
        date_i = today - timedelta(days=i)
        count = sum(datetime.strptime(alert.date, '%Y-%m-%d').date() == date_i for alert in fire_alerts)
        data.append({'date': date_i.strftime('%Y-%m-%d'), 'count': count})

    return jsonify(data)


@View.route('/store_location', methods=['POST'])
def store_location():
    now = datetime.now()
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')

    # Check if latitude and longitude values are valid
    if latitude is None or longitude is None:
        return jsonify({'success': False, 'error': 'Invalid latitude or longitude'})

    # Create a new Fire_Location object
    new_location = Fire_Location(date=str(now.date()), time=str(now.time()), latitude=latitude, longitude=longitude)

    # Add the location to the database
    db.session.add(new_location)
    db.session.commit()

    return jsonify({'success': True})


@View.route('/ModelTesting')
def ModelTesting(): 
    history=np.load('my_history.npy',allow_pickle='TRUE').item()
    # set the font globally
    params = {'font.family':'Comic Sans MS',
              "xtick.color" : "white",
              "ytick.color" : "white"}
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(8,4)) 
    fig.patch.set_facecolor('xkcd:mahogany')

    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy', fontsize=16, color='white')
    plt.ylabel('Accuracy', fontsize=14, color='white')
    plt.xlabel('Epochs', fontsize=14, color='white')
    plt.legend()
    plt.savefig('static/graphs/Acc_plot.png', bbox_inches='tight', pad_icnhes=0)
    plt.close()

    fig2 = plt.figure(figsize=(8,4)) 
    fig2.patch.set_facecolor('xkcd:mahogany')
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss', fontsize=16, color='white')
    plt.ylabel('Loss', fontsize=14, color='white')
    plt.xlabel('Epochs', fontsize=14, color='white')
    plt.legend()
    plt.savefig('static/graphs/Loss_plot.png', bbox_inches='tight', pad_icnhes=0)
    plt.close()

    return render_template('ModelTesting.html', acc_plot_url="static/graphs/Acc_plot.png", loss_plot_url="static/graphs/Loss_plot.png")


@View.route('/delete_alert/<int:alert_id>', methods=['POST'])
def delete_alert(alert_id):
    alert = Fire_Alerts.query.get_or_404(alert_id)
    try:
        db.session.delete(alert)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@View.route("/Prediction", methods=['GET', 'POST'])
def Prediction():
    return render_template("Prediction.html")


@View.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/uploads/" + img.filename
        img.save(img_path)

        p = predict_label(img_path)
    return render_template("Prediction.html", prediction=p, img_path=img_path)


@View.route('/Location')
def Location():
    return render_template('Location.html')


@View.route('/LiveMonitor')
def LiveMonitor():
    return render_template('LiveMonitor.html')


@View.route('/video_feed')
def video_feed():
    app = create_app()  # Create the Flask app object
    return Response(gen_frames(app), mimetype='multipart/x-mixed-replace; boundary=frame')


@View.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('neg') == 'NEGATIVE':
            global neg
            neg = not neg
        elif request.form.get('stop') == 'MONITOR':
            if switch == 0:
                camera = cv2.VideoCapture(0)
                switch = 1
            else:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
        elif request.form.get('rec') == 'RECORD':
            global rec, out
            rec = not rec
            if rec:
                now = datetime.now()
                video_folder = 'static/video/'
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('static/video/vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif rec == False:
                out.release()
    elif request.method == 'GET':
        return render_template('LiveMonitor.html')
    return render_template('LiveMonitor.html')

