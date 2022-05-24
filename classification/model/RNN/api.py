import json
import pickle
import torch
import pickle

from typing import Type
from flask import request
from flask import Flask, render_template, request, url_for, redirect, session, flash

import dataset

# 1. 모델저장 2. vocab 저장 3. 불러와서 integer encoding 4. predict 5. 결과값 전송


app = Flask(__name__, static_url_path='/static')

@app.route('/predict', method=['GET', 'POST'])
def index():
    params = json.loads(request.get_data(), encoding='utf-8')
    
    input_text = params['text']
    return None