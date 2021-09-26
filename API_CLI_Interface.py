#!/bin/bash/python3

import requests

while(True):
    query = {'input':input("You: ")}
    response = requests.put('http://3.84.125.186:5000', params=query)
    print("Cogswell: ",response.json()['body'])