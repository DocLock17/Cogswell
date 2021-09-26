#!/bin/bash/python3

import requests

while(True):
    query = {'input':input("You: ")}
    # response = requests.put('http://3.235.0.255:5000', params=query)
    response = requests.put('http://67.202.54.165:5000', params=query)
    print("Cogswell: ",response.json()['body'])