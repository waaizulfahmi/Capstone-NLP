from flask import Flask, render_template, request
import pickle

# filename = 'Apps_Classification.pkl'
classifier = pickle.load(open('Apps_Classification.pkl', 'rb'))
cv = pickle.load(open('count-Vectorizer.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = classifier.predict(vect)
    return render_template('index.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
