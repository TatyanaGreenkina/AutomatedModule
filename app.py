import json
from flask import Flask, request, abort, render_template
import NeuralNet.CNN as cnn
from rankin import Ranking

rank = Ranking()
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
    return json.dumps(task, ensure_ascii=False), 201


@app.route('/main', methods=['GET', 'POST'])
def search():
    elems = []
    answers = prep.rows
    if request.method == 'POST':
        message = request.form['text']
        for i in rank.search_simil_elems(message, 5):
            elems.append('-----'.join([i[0], str(i[1])]))
    return render_template('add_message.html', answers=answers, elems=elems)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# s = 'болт насосы гайка'
# # print(rank.search_simil_elems(s, 5))
# for i in rank.search_simil_elems(s, 5):
#     print('-----'.join([i[0], str(i[1])]))

