import json
import pickle
import torch
import pickle

from BERT import BERTDataset, BERTClassifier
from kobert.pytorch_kobert import get_pytorch_kobert_model
from typing import Type
from flask import request
from flask import Flask, render_template, request, url_for, redirect, session, flash


app = Flask(__name__, static_url_path='/static')


@app.route('/predict', methods=['GET', 'POST'])
def index():
    params = json.loads(request.get_data(), encoding='utf-8')

    # Flow
    input_text = params["text"]
    test_set = BERTDataset(input_text, [0]*len(input_text), tokenizer, max_len, True, False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    kobert_model, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(kobert_model, num_classes=len(label_enc.classes_), dr_rate=dr_rate).to(device)
    model.load_state_dict(torch.load(model_dir))

    pred_y = []
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
        output = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
        _, output = torch.max(output, 1)
        output = output.tolist()
        output = label_enc.inverse_transform(output)
        pred_y.append(output[0])
    
    return json.dumps(pred_y, ensure_ascii=False)


if __name__ == '__main__':
    # Directory Setting
    model_dir = "model_kobert.pt"
    tokenizer_dir = "tokenizer_kobert.pickle"
    label_enc_dir = "label_enc_kobert.pickle"

    # Setting Parameter
    device = torch.device("cuda")

    # Hyper Parameter
    max_len = 10
    dr_rate = 0.5

    tokenizer = pickle.load(open(tokenizer_dir, 'rb'))
    label_enc = pickle.load(open(label_enc_dir, 'rb'))

    app.run(host='0.0.0.0', port='1001', debug=True)