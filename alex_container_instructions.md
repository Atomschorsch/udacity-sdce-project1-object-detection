# Docker instructions and setup
Here you can find instructions on how to build and run the docker container.
I have tried quite some different methods
- Running it directly on windows
- Runnin it on docker desktop on windows host
- Running it in docker on wsl on windows host

I had some difficulties to show the plots created by matplotlib from within docker. The necessary adaptions can be seen in this file and also in the Dockerfile.
The final setup to use is now docker on wsl2 on windows 11 pro host.

Mounted folders:
- data -> /mnt/data
- repo -> /app/project

Mapped ports:
- 6006:
Use Tensorboard outside of the container via localhost:6006

Fixes to be able to run all this are entered to the Dockerfile and taken from udacity knowledge center:
```bash
# ALEX: My own adaptions to make it run on my system and to use matplotlib on windwos host together with XLaunch
# Fix from https://knowledge.udacity.com/questions/889534
RUN pip install keras==2.5.0rc
# Fix for using matplotlib in docker
RUN apt-get install -y python3-tk
```


## Build docker
1. Goto directory where the docker file is contained**
```bash
docker build -t project-dev -f Dockerfile .
```

## Run Docker in WSL on Windows host
```bash
docker run --gpus all -p 6006:6006 -e DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v /usr/lib/wsl:/usr/lib/wsl --device=/dev/dxg -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e PULSE_SERVER=$PULSE_SERVER  --mount "source=/home/alex/repos/udacity/project1_object_detection/Data/,target=/mnt/data,type=bind" --mount  "source=/home/alex/repos/udacity/project1_object_detection/udacity-sdce-project1-object-detection/,target=/app/project,type=bind" --name udacity_project1 --network=host -ti project-dev bash
```


## GUI in docker in WSL
See https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md
for GUIs in Container:
```bash
-v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -e WAYLAND_DISPLAY  -e XDG_RUNTIME_DIR -e XDG_RUNTIME_DIR -e PULSE_SERVER
```
for GUIs in Container with GPU:
```bash
-v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v /usr/lib/wsl:/usr/lib/wsl --device=/dev/dxg -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e PULSE_SERVER=$PULSE_SERVER --gpus all
```

## Python adaptions necessary to see matplotlib outside of container
```python
import matplotlib
import tkinter
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
```