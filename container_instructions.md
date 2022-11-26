# Build Docker

# 1. Goto directory where the docker file is contained
docker build -t project1-dev -f Dockerfile .


# Run Docker
# use display variable only on win 10, win 11 is already working
export DISPLAY=192.168.198.34:0.0
winpty docker run --gpus all -p 6006:6006 -e DISPLAY=$DISPLAY -v /c/Repos/Udacity/project1/nd013-c1-vision-starter/:/app/project/ --mount "source=/c/Repos/Udacity/project1/data/,target=/mnt/data,type=bind" --mount "source=/c/Repos/Udacity/project1_submission/nd013-c1-vision-starter,target=/app/project,type=bind" --name udacity_project1_submission --network=host -ti project1-dev bash