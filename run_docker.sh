docker run -it \
    --name="hdrseg" \
    --gpus=all \
    --shm-size=32g \
    --volume="$(pwd):/home/app/HDRSeg-UDA" \
    hdrseg /bin/bash