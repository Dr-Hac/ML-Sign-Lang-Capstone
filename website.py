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

# set app configs
app = Flask(__name__)
app.config['SECRET_KEY'] = 'key'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
# define file extension type to [.jpg, .jpe, .jpeg, .png, .gif, .svg, and .bmp]
photos = UploadSet('photos', extensions=IMAGES)
configure_uploads(app, photos)


# create upload form
class UploadForm(FlaskForm):
    # defines multiple inputs
    photo = MultipleFileField('File(s) Upload')
    # adds Upload button
    submit = SubmitField('Upload')


def on_close():
    print('app is closing')
    # cleanup uploaded files on close
    os.system('del "C:\\Users\\caida\\PycharmProjects\\not-hotdog\\uploads\\*" /q /s')


# define route to place uploaded files
@app.route('/uploads/<filename>')
def get_file(filename):
    # gets file from /uploads/<filename>
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


# set the website to run on loopback and use GET, POST methods
@app.route('/', methods=['GET', 'POST'])
def home():
    # define variables
    global file_url
    not_hotdog_data = None
    form = UploadForm()
    url_array = []
    # check that form is both valid and submitted
    if form.validate_on_submit():
        # reset variables for new inputs
        url_array = []
        filenames = []
        # loop though all uploaded files
        for file in form.photo.data:
            # capture the filename
            filename = secure_filename(file.filename)
            # save the file to /uploads/<filename>
            file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
            # create an array of file names
            filenames.append(filename)
        # reset variables for new inputs
        classification_array = []
        confidence_array = []
        # loop through all filenames
        for file in filenames:
            # create url for file
            file_url = url_for('get_file', filename=file)
            # create array of urls
            url_array.append(file_url)
            print(file_url)
            # run the image through the not_hotdog model ([1:] is needed to remove the first / of the input)
            not_hotdog_data = not_hotdog.run(file_url[1:])
            # create arrays of both outputs from not_hotdog.py
            classification_array.append(not_hotdog_data[0])
            confidence_array.append(str(not_hotdog_data[1] * 100)[:5])
        print(url_array)
        # send the form, urls, classifications, and confidence scores to E.html
        return render_template('HTML.html', form=form, url_array=url_array, classification=classification_array, confidence=confidence_array)
    else:
        url_array = []
        url_array.append(None)
        # send original values before any upload to E.html
        return render_template('HTML.html', form=form, url_array=url_array, classification=['no image uploaded'], confidence=[0])


if __name__ == '__main__':
    # regisgter the on_close() function with the flask app
    # atexit.register(on_close)
    # run the flask app
    app.run(debug=True)
