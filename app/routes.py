from flask import Flask
from flask import Flask, Response, request, jsonify, render_template
from file_read_backwards import FileReadBackwards

from app import app
from app.model import CNNLSTMClass
from app.forms import RunAnalysisForm

# app = Flask(__name__)

# def file_test():
#     """
#     file name of test datd
#     """
#     if len(sys.argv) == 1:
#         FILE_TEST = None
#     else:
#         FILE_TEST = sys.argv[1]
#         file      = int(FILE_TEST[9])
#         print("test data file name:", FILE_TEST)
#     return FILE_TEST

@app.route('/', methods=['GET', 'POST'])
def home():
    form = RunAnalysisForm()
    if form.validate_on_submit():
        file_test = form.engine_no.data
        if request.method == 'POST':
            index_names = ['unit_nr', 'time_cycles']
            setting_names = ['setting_1', 'setting_2', 'setting_3']
            sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
            col_names = index_names + setting_names + sensor_names
            test_data = pd.read_csv(request.files.get('file'), sep='\s+', header=None, 
                    names=col_names)

        report = CNNLSTMClass.CNNLSTM(dataset="cmapss", file=test_data, file_test=file_test, Train=FALSE, trj_wise=True, plot=True)

    return render_template('data_upload.html', form=form, title='Turbofan Engine Analysis: Predictive Maintenance')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
