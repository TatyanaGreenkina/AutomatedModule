from flask import Flask, render_template, request, url_for, redirect
import embeddings

app = Flask(__name__)

embeding_matrix = embeddings.Embedding()
answers = []


@app.route('/main', methods=['GET'])
def main():
    return render_template('add_message.html', answers=answers)


@app.route('/add_message', methods=['POST'])
def add_message():
    message = request.form['text']
    answers.clear()
    words_dict = embeding_matrix.search_words(message)
    for key in words_dict:
        answers.append(key + '          ' + str(words_dict[key]))
    return redirect(url_for('main'))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
