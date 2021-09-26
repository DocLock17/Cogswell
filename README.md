# Cogswell
A simple chatbot API for presentation at Cognizant Technologies


## Move in to the application directory
cd DockerSpeechAPI

## To build your application
docker image build --tag doclock17/cogswell:1.0 .

## To Start your Application
docker container run --d --p 5000:5000 --n cogswell doclock17/cogswell:1.0

## To Check on Running Containers
docker ps -a 
docker container ls --all

## To Force Close Running Containers
docker container rm --force cogswell
docker rm $(docker ps --filter status=exited -q)
docker image prune
