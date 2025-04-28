docker run -it \
    --name="hdrseg" \
    --gpus=all \
    --shm-size=32g \
    --volume="$(pwd):/home/app/RMSeg-HDR" \
    hdrseg /bin/bash