import numpy as np
import pandas as pd
import logging
import re
import os 
import base64
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
            example_engines_df = pd.read_csv(request.files.get('file'), sep='\s+', header=None, 
                    names=col_names)
            file_no = int(list(file_test)[9])
            engine_no = [str(i) for i in list(file_test)[12:15]]
            engine_no = "".join(engine_no)
            engino_no = int(engine_no)
            flash("Done!")

        report = CNNLSTMClass.CNNLSTM(dataset="cmapss", file_no=file_no, file_test=file_test, Train=False, trj_wise=True, plot=True)

        filename = os.getcwd()+"/app/static/images/rul.png"
        try:
            with open(filename, 'rb') as file:
                returnfile = file.read()

            return Response(returnfile, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                headers={"Content-disposition": f"attachment; filename=Engine {engine_no} rul.png"})
        except Exception as e:
            logging.error(str(e))
        
        try:
            return redirect(next_page)
        except Exception as e:
            logging.error(str(e))

    return render_template('index.html', form=form, title='Turbofan Engine Analysis: Predictive Maintenance')

@app.route('/plot')
def plot():
    return render_template('plot.html', url=f'/static/images/rul.png')

# @app.route('/index')
# def template(): 
#     return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
