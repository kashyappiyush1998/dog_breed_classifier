import json
import os
from flask import Flask
from flask import render_template, request, jsonify
from scripts import predict

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/upload-image", methods=["POST"])
def uploadImage():
    imagefile = request.files['file']
    filepath = os.path.join(os.path.dirname(__file__) + "/app/image", 'image.jpg')
    print(imagefile)
    imagefile.save(filepath)

    msg = predict.predict_breed(filepath)
    print(msg)
    return msg


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()