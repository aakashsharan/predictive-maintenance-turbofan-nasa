import numpy as np
import pandas as pd
import logging
import re
import os 
from flask import Flask
from flask import Flask, Response, request, jsonify, render_template, flash, send_file
from file_read_backwards import FileReadBackwards

from app import app
from app.model import CNNLSTMClass
from app.forms import RunAnalysisForm


@app.route('/', methods=['GET', 'POST'])
def home():
    form = RunAnalysisForm()
    if form.validate_on_submit():
        if request.method == 'POST':
            index_names = ['unit_nr', 'time_cycles']
            setting_names = ['setting_1', 'setting_2', 'setting_3']
            sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
            col_names = index_names + setting_names + sensor_names
            file = request.files.get('file')
            file_test = file.filename
            logging.warning("routes file_test: " + str(file_test))
            example_engines_df = pd.read_csv(request.files.get('file'), sep='\s+', header=None, 
                    names=col_names)
            # test_engine_id = example_engines_df['unit_nr']
            logging.warning("test_engine_id.shape: " + str(test_engine_id.shape))
            file_no = int(list(file_test)[9])
            logging.warning("file_no: " + str(file_no))
            flash("Done!")

        report = CNNLSTMClass.CNNLSTM(dataset="cmapss", file_no=file_no, file_test=file_test, Train=False, trj_wise=True, plot=True)

        # response = report.run()

    return render_template('data_upload.html', form=form, title='Turbofan Engine Analysis: Predictive Maintenance')

@app.route('/download')
def downloadFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = os.getcwd()+"/app/static/images/rul.png"
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
