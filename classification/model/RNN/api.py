import os
import json
import torch
import pickle

from typing import Type
from flask import request
from flask import Flask, render_template, request, url_for, redirect, session, flash

import utils
import rnn

app = Flask(__name__, static_url_path='/static')

# load lately trained model
model_file_name = os.listdir('checkpoint')
model_state_dict = torch.load(os.path.join('checkpoint', model_file_name))
model = rnn.SimpleRNN(n_classes=2,
                      vocab_size=10000,
                      embedding_dim=32,)
model = model.load_state_dict(model_state_dict)


@app.route('/predict', method=['GET', 'POST'])
def index():
    params = json.loads(request.get_data(), encoding='utf-8')
    input_text = params['text']
    
    # load vocabulrary object
    vocab_file = os.listdir('vocab')[0]
    vocab_obj = torch.load(os.path.join('vocab', vocab_file))
    
    # integer encoding
    tokenizer = utils.get_tokenizer('Kkma')
    input_sequence = utils.integer_encoding(input_text, tokenizer, vocab_obj)
   
    # predict
    output = model(input_sequence)
    pred = output.max(1).indices
    
    return json.dumps(pred, ensure_ascii=False)