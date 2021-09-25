#!/usr/bin/env python

from flask import Flask
from flask import request
from flask import jsonify

from ChatNet import IntentModel

# Instatiate flask server
app = Flask(__name__)
cNet = IntentModel()

# Set up one endpoint
@app.route('/', methods=['GET'])
def landing():
    # Return JSON body
    return """Welcome to the Cogswell REST API. Submit a PUT request using the 'input' \
    KEY and a string VALUE containing a user query in plain language and the Cogswell bot response will be returned as the VALUE of the 'body' KEY.""", 200

# Final PUT endpoint
@app.route('/', methods=['PUT'])
def string_flip():
    print("request received")
    put_input = request.args.get('input')
    print(put_input)

    try:
        # Get response
        ints = cNet.predict_class(put_input, cNet.model)
        body_output = cNet.get_response(ints, cNet.intents)
        print(body_output)
    except Exception as e:
        print(e)
    # except:
    #     body_output = 'Sorry API seems to have experianced an error'
    #     print(body_output)
    return jsonify(body=body_output), 200

# Run Code
if __name__ == '__main__':
    app.run(host="0.0.0.0")
    # app.run()