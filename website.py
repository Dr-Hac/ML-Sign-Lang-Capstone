"""
NAME: website.py
DESCRIPTION: create and run the webcam on a website at the localhost/5000 address
PROGRAMMER: Caidan Gray
CREATION DATE: 4/7/2025
LAST EDITED: 4/8/2025   (please update each time the script is changed)
"""

import os
from flask import Flask, send_from_directory, url_for, render_template, request, redirect
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import MultipleFileField, SubmitField
import atexit

import webcam_link

# set app configs
app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'

webcam = webcam_link.Webcam_Link(0)


def on_close():
    print('app is closing')
    # cleanup uploaded files on close
    # os.system('del "C:\\Users\\caida\\PycharmProjects\\not-hotdog\\uploads\\*" /q /s') (fix this for generalized use)



# set the website to run on loopback and use GET, POST methods
@app.route('/', methods=['GET', 'POST'])
def home():
    frame = webcam.get_frame()
    print(frame)
    return render_template('HTML.html', zero=0, url=frame, frame=frame)


if __name__ == '__main__':
    # register the on_close() function with the flask app
    atexit.register(on_close)
    # run the flask app
    app.run(debug=True)
