import csv
import io

import onnxruntime as rt
import pandas as pd
from argparse import ArgumentParser
from flask import Flask, request, jsonify
from predict import predict_batch, predict_url, evaluate
import constants
from crawling import save_results_and_texts, get_links


def parse_args():
    parser = ArgumentParser('ML web app')
    parser.add_argument('--model',
                        type=str,
                        help='model to use')
    parser.add_argument('--port',
                        type=int,
                        default=8080,
                        help='Port to use')
    parser.add_argument('--host',
                        type=str,
                        default='0.0.0.0',
                        help='Host to use')
    return parser.parse_args()


args = parse_args()
sess = rt.InferenceSession(args.model)
app = Flask('Test app')


@app.route('/model/metadata')
def meta():
    metadata_map = sess._model_meta.custom_metadata_map
    metadata_json = jsonify(metadata_map)
    return metadata_json


@app.route('/model/forward', methods=['GET', 'POST'])
def predict():
    try:
        url = request.get_json()['url']
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    return jsonify(predict_url(url, sess))


def extract_df(content):
    stream = io.StringIO(content.stream.read().decode("UTF-8"), newline=None)
    csv_input = csv.reader(stream)
    df = pd.DataFrame(csv_input)
    columns = df.loc[0, :].tolist()
    data = df.loc[1:, :].to_numpy()
    df = pd.DataFrame(data, columns=columns)
    df[constants.FIELD_IS_TEXT] = df[constants.FIELD_IS_TEXT].astype(int)
    return df


@app.route('/model/forward_batch', methods=['GET', 'POST'])
def forward_batch():
    try:
        content = request.files['datafile']
        df = extract_df(content)
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    res = predict_batch(df, sess)
    return jsonify(res[0]), res[1]


@app.route('/model/evaluate', methods=['GET', 'POST'])
def evaluate_batch():
    try:
        content = request.files['datafile']
        df = extract_df(content)
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    res = evaluate(df, sess)
    return jsonify(res[0]), res[1]


@app.route('/process_link', methods=['GET', 'POST'])
def process_url():
    try:
        link = request.get_json()['url']
    except KeyError:
        return 'bad request', constants.CODE_BAD_REQUEST
    links = get_links(link)
    results = []
    for link in links:
        results.append(predict_url(link, sess))
    found_text_cnt, saved_text_cnt = save_results_and_texts(links, results, link)
    return jsonify({'Links': len(links), 'Texts found': found_text_cnt, 'Texts saved': saved_text_cnt})


if __name__ == '__main__':
    app.run(port=args.port, host=args.host)
