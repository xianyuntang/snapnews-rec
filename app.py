from flask import Flask, request
from model import  VGGATTModel
import json
import numpy as np
from tensorflow.python.keras import backend as K
import tqdm
import cv2
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
app = Flask(__name__)

app.config['SECRET_KEY'] = os.urandom(24)


def labels_to_text(charset, label):
    ret = ''
    for l in label:
        if l == -1:  # CTC Blank
            continue
        else:
            ret += charset[l]

    return ret


def decode_predict_ctc(out, top_paths=1):
    def labels_to_text(label):
        ret = []
        for batch_label in label:
            tmp = []
            for l in batch_label:
                if l == -1:  # CTC Blank
                    tmp.append('')
                else:
                    tmp.append(charset[l])

            ret.append(''.join(tmp))
        return ret

    charset = u''
    # print("####### loading character ######")
    with open(charset_file, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f.readlines()):
            charset += line.replace('\n', '')
    results = []
    beam_width = 10
    print(len(charset)+1)
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        labels = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])
        text = labels_to_text(labels)
        results.append(text)
    return results


def predict(image):
    height, width, depth = image.shape
    # 直的
    if height > width:
        new_height = int(32 * (height / width))
        image = cv2.resize(image, (32, new_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
        image = image.astype(np.float32)
        image = image / 255
    else:
        new_width = int(width / height * 32)
        image = cv2.resize(image, (new_width, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)

        image = image.astype(np.float32)
        image = image / 255

        image = image.T
    image = np.expand_dims(image, -1)
    image = np.expand_dims(image, 0)
    predictions = ocr_model(image)
    return decode_predict_ctc(predictions)[0][0]


K.set_learning_phase(0)
ocr_model = VGGATTModel(training=False)
ocr_model(np.random.random(size=(1, 320, 32, 1)).astype(np.float32))
ocr_model.load_weights(r"./static/finetune_model-ckpt-35000-loss-2.989916.h5")
charset_file = r'./static/charset.txt'


@app.route('/POST/recognition', methods=['POST'])
def recognition():
    if request.method == 'POST':
        if 'file' not in request.files:
            pass
        file = request.files['file'].read()
        height = int(request.files['height'].read().decode('utf-8'))
        width = int(request.files['width'].read().decode('utf-8'))
        depth = int(request.files['depth'].read().decode('utf-8'))
        image = np.fromstring(file, dtype=np.uint8)
        image = np.reshape(image, (height, width, depth))
        result = predict(image)
        return json.dumps(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
