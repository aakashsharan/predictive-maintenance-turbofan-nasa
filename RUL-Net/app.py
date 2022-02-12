import flask
from flask import Flask, request, jsonify, render_template
from file_read_backwards import FileReadBackwards

app = flask.Flask(__name__)

with FileReadBackwards("app.py") as f:
    # getting lines by lines starting from the last line up
    b_lines = [ row for row in f ]

@app.route('/', methods=['GET'])
def home():
    return render_template('example.html', b_lines=b_lines)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
