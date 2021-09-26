#!/bin/bash/python3

import requests

while(True):
    query = {'input':input("You: ")}
    # response = requests.put('http://3.235.0.255:5000', params=query)
    response = requests.put('https://67.202.54.165:5000', params=query)
    # response = requests.put('http://ec2-3-235-0-255.compute-1.amazonaws.com:5000', params=query)
    print("Cogswell: ",response.json()['body'])