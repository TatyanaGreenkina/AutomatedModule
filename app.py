import json
from flask import Flask, request, abort
import NeuralNet.CNN as cnn


cnn_net, prep, index_to_count = cnn.prom_cnn()

app = Flask(__name__)


@app.route('/add_message', methods=['POST'])
def add_message():
    if not request.json or not 'str' in request.json:
        abort(400)
    message = request.json['str']
    prep_message = prep.prepare_data(message)
    if prep_message == '<UNK>':
        count = 'NOUN'
    else:
        batch = [[prep.word_to_index[i] for i in prep_message]]
        count = index_to_count[cnn_net.predict(tok=batch)[0]]
    task = {
        'str': message,
        'count': count
    }
    # answers.clear()
    # answers.append(prep.searcher(message))
    return json.dumps(task, ensure_ascii=False), 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
