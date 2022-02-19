import pandas as pd
import logging
import re
from flask import Flask
from flask import Flask, Response, request, jsonify, render_template, flash
from file_read_backwards import FileReadBackwards

from app import app
from app.model import CNNLSTMClass
from app.forms import RunAnalysisForm


@app.route('/', methods=['GET', 'POST'])
def home():
    logging.warning("initialization works")
    form = RunAnalysisForm()
    if form.validate_on_submit():
        logging.warning("form validated")
        if request.method == 'POST':
            logging.warning("method is post")
            # index_names = ['unit_nr', 'time_cycles']
            # setting_names = ['setting_1', 'setting_2', 'setting_3']
            # sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
            # col_names = index_names + setting_names + sensor_names
            file = request.files.get('file')
            file_test = file.filename
            # test_data = pd.read_csv(request.files.get('file'), sep='\s+', header=None, 
            #         names=col_names)
            # logging.warning(str(list(request.files.get('file'))))
            logging.warning(str(file_test))
            file_no = int(list(file_test)[9])
            logging.warning("file_no is " + str(file_no))

        report = CNNLSTMClass.CNNLSTM(dataset="cmapss", file_test=file_test, file_no=file_no, Train=False, trj_wise=True, plot=True)

        response = report.run()

    return render_template('data_upload.html', form=form, title='Turbofan Engine Analysis: Predictive Maintenance')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
