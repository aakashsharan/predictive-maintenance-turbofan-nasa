import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect, CSRFError

app = Flask(__name__)

from app import routes

csrf = CSRFProtect(app)


SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY