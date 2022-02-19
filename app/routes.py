import numpy as np
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
    form = RunAnalysisForm()
    if form.validate_on_submit():
        if request.method == 'POST':
            index_names = ['unit_nr', 'time_cycles']
            setting_names = ['setting_1', 'setting_2', 'setting_3']
            sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
            col_names = index_names + setting_names + sensor_names
            file = request.files.get('file')
            file_test = file.filename
            example_engines_df = pd.read_csv(request.files.get('file'), sep='\s+', header=None, 
                    names=col_names)
            test_engine_id = np.unique(example_engines_df['unit_nr'])
            file_no = int(list(file_test)[9])
            flash("Done!")

        report = CNNLSTMClass.CNNLSTM(dataset="cmapss", test_engine_id=test_engine_id, example_engines_df= example_engines_df, file_test=file_test, file_no=file_no, Train=False, trj_wise=True, plot=True)

        # response = report.run()

    return render_template('data_upload.html', form=form, title='Turbofan Engine Analysis: Predictive Maintenance')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
