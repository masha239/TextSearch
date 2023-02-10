import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import constants
from extract import process_url


def predict(X, sess):
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name],
                        {input_name: X.astype(np.float32)})
    return pred_onx[0]


def predict_url(url, sess):
    code, features = process_url(url)
    if code != constants.CODE_OK:
        return 'Model failed', constants.CODE_FAILED
    prediction = predict(np.array([features]), sess)[0]
    return {'answer': int(prediction)}, constants.CODE_OK


def predict_urls(urls, sess):
    X = []
    ok = []

    for url in urls:
        answer = process_url(url)
        if answer[0] == constants.CODE_OK:
            X.append(answer[1])
            ok.append(True)
        else:
            ok.append(False)

    X = np.array(X)
    y_pred = predict(X, sess)
    length = len(urls)
    answer = []
    j = 0
    for i in range(length):
        if ok[i]:
            answer.append(y_pred[j])
            j += 1
        else:
            answer.append(None)

    return answer


def make_str(arr):
    return [str(x) for x in arr]


def predict_batch(df, sess):
    answer = predict_urls(df.url, sess)
    return {'answer': make_str(answer)}, constants.CODE_OK


def calculate_metrics(y_pred, y_test):
    return {'accuracy': accuracy_score(y_pred, y_test),
            'f1': f1_score(y_pred, y_test)}


def evaluate(df, sess):
    y_test = df[constants.FIELD_IS_TEXT].tolist()
    y_pred = predict_urls(df.url, sess)
    y1 = [y_pred[i] for i in range(len(y_pred)) if y_pred[i] is not None]
    y2 = [y_test[i] for i in range(len(y_pred)) if y_pred[i] is not None]
    if len(y1) == 0:
        return {'answer': make_str(y_pred), 'metrics': {}}, constants.CODE_FAILED
    metrics = calculate_metrics(y1, y2)
    return {'answer': make_str(y_pred), 'metrics': metrics}, constants.CODE_OK
