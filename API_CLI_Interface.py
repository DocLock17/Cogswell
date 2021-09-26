#!/bin/bash/python3

import requests

while(True):
    query = {'input':input("You: ")}
    response = requests.put('http://54.175.97.43:5000', params=query)
    print("Cogswell: ",response.json()['body'])