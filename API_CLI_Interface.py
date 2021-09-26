#!/bin/bash/python3

import requests

while(True):
    query = {'input':input("You: ")}
    response = requests.put('http://35.188.177.166:5000', params=query)
    print("Cogswell: ",response.json()['body'])