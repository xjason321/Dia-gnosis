from predict import *
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    # Main website
    return render_template('index.html')

if __name__ == '__main__':
    app.run()