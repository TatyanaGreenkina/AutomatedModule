import json
from flask import Flask, request, abort
import preprocessing
import NeuralNet.CNN as cnn

prep = preprocessing.Preprocessing(r'dict', special_token='<UNK>')

app = Flask(__name__)


@app.route('/add_message', methods=['POST'])
def add_message():
    if not request.json or not 'str' in request.json:
        abort(400)
    message = request.json['str']
    task = {
        'str': message,
        'count': cnn.prom_cnn(message)
    }
    # answers.clear()
    # answers.append(prep.searcher(message))
    return json.dumps(task, ensure_ascii=False), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
