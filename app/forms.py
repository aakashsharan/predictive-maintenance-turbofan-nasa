import wtforms
from flask_wtf import FlaskForm as Form
from wtforms import SelectField, SubmitField

from wtforms.validators import DataRequired

class RunAnalysisForm(Form): 
    engine_no = SelectField("Engine Number", choices=[
        ('1', 1), 
        ('2', 2), 
        ('3', 3)
    ])
    submit = SubmitField('Run Analysis')