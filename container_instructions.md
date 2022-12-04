# Build Docker

# 1. Goto directory where the docker file is contained
docker build -t project-dev -f Dockerfile .

# Run Docker in WSL on Windows host
docker run --gpus all -p 6006:6006 -e DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v /usr/lib/wsl:/usr/lib/wsl --device=/dev/dxg -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e PULSE_SERVER=$PULSE_SERVER  --mount "source=/home/alex/repos/udacity/project1_object_detection/Data/,target=/mnt/data,type=bind" --mount  "source=/home/alex/repos/udacity/project1_object_detection/udacity-sdce-project1-object-detection/,target=/app/project,type=bind" --name udacity_project1 --network=host -ti project-dev bash


# GUI in docker in WSL: https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md
# für GUIs in Container: -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -e WAYLAND_DISPLAY  -e XDG_RUNTIME_DIR -e XDG_RUNTIME_DIR -e PULSE_SERVER
# für GUIs in Container und GPU: -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v /usr/lib/wsl:/usr/lib/wsl --device=/dev/dxg -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e PULSE_SERVER=$PULSE_SERVER --gpus all

## Python adaptions necessary to see matplotlib outside of container
# import tkinter
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt