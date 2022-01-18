#!/bin/bash
docker build --force-rm --shm-size=64g -t u_20_cuda_11_5_0_torch_1_11_trt_8_2_1_8 -f ./docker_file/Dockerfile .
bash ./docker_run_gui_mode_on_remote_server_from_local_server_with_option_1_image_name_and_tag_option_2_3_N_shared_dirs.sh u_20_cuda_11_5_0_torch_1_11_trt_8_2_1_8 ~/work/etc/colorful_colorization_pytorch ~/data/tiny-imagenet-200
