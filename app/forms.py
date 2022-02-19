import wtforms
from flask_wtf import FlaskForm as Form
from wtforms import SelectField, SubmitField

from wtforms.validators import DataRequired

class RunAnalysisForm(Form): 
    submit = SubmitField('Run Analysis')