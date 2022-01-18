#!/bin/bash
# usage : $ bash docker_run_gui_mode_on_remote_server_from_local_server_with_option_1_container_name_and_tag_option_2_shared_dir.sh my_image:my_tag /path/to/shared_dir_1 /path/to/shared_dir_2 ...... /path/to/shared_dir_N
echo "command line : $@"
echo "# of arguments : $#"
IMAGE_AND_TAG=$1
echo "IMAGE_AND_TAG : $IMAGE_AND_TAG"
for (( i = 2; i <= $#; i++ )); do
    #echo "Planet #$i is ${!i}"
    SHARED_DIR_HOST="${!i}"
    SHARED_DIR_CONTAINER=`basename "$SHARED_DIR_HOST"`
    SHARED_VOLUME_LIST="-v $SHARED_DIR_HOST:/workspace/$SHARED_DIR_CONTAINER $SHARED_VOLUME_LIST"
done
echo "SHARED_VOLUME_LIST : " $SHARED_VOLUME_LIST
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth-n
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH
#docker run -ti -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --net host $1 /bin/bash
#docker run -ti --gpus all --rm --privileged -e DISPLAY=$DISPLAY -v $SHARED_DIR_HOST:/root/$SHARED_DIR_CONTAINER -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --net host $IMAGE_AND_TAG /bin/bash -c "apt-get update; apt-get install fish gedit -y; (gedit &); fish"
#docker run -ti --gpus all --rm --privileged -e DISPLAY=$DISPLAY $SHARED_VOLUME_LIST -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --net host $IMAGE_AND_TAG /bin/bash -c "apt-get update; apt-get install fish gedit -y; (gedit &); fish"
docker run --ipc=host -ti --gpus all --rm --privileged -e DISPLAY=$DISPLAY $SHARED_VOLUME_LIST -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --net host $IMAGE_AND_TAG /bin/bash -c "apt-get update; apt-get install fish gedit -y; (gedit &); fish"
#
#docker run -ti --gpus all --rm --privileged -e DISPLAY=$DISPLAY -v $SHARED_DIR_HOST:/root/$SHARED_DIR_CONTAINER -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH --net host $IMAGE_AND_TAG /bin/bash -c "apt-get update; apt-get install fish gedit -y; (gedit &); killall -9 gedit; fish"
