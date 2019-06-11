# This docker is set up to run habitat challenge but also has everything we needed so far

# docker run --runtime=nvidia -ti --rm -v  /share/home/habitat_data:/root/perception_module/habitat-api/data habitatsubmission:latest
# CHALLENGE_CONFIG_FILE=/root/perception_module/habitat-api/configs/tasks/pointnav_gibson_val_mini.yaml ./submission.sh

#FROM nvidia/cudagl:9.0-base-ubuntu16.04
FROM fairembodied/habitat-challenge:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/cuda/samples/5_Simulations/nbody

RUN make

#CMD ./nbody

RUN apt-get update && apt-get install -y curl  && apt-get install -y apt-utils
RUN conda update -y conda

RUN . activate habitat
ENV PATH /opt/conda/envs/habitat/bin:$PATH

###############################
#  set up habitat
###############################
WORKDIR /root
RUN git clone https://github.com/alexsax/midlevel-reps.git

WORKDIR /root/midlevel-reps
RUN git checkout v1.0 && rm -r habitat-api && rm -r habitat-sim
RUN git clone https://github.com/facebookresearch/habitat-sim.git
RUN git clone https://github.com/facebookresearch/habitat-api.git

WORKDIR /root/midlevel-reps/habitat-sim
RUN conda install -y cmake
RUN pip install numpy
RUN python setup.py install --headless

WORKDIR /root/midlevel-reps/habitat-api
RUN git checkout 05dbf7220e8386eb2337502c4d4851fc8dce30cd
RUN pip install --upgrade -e .
RUN apt-get install -y ffmpeg
ADD habitat_data /root/midlevel-reps/habitat-api/data
RUN rm -r /root/midlevel-reps/habitat-api/configs
ADD habitat_configs /root/midlevel-reps/habitat-api/configs
RUN rm -r baselines

###############################
#  set up midlevel-reps
###############################
ADD taskonomy_models /mnt/models
ADD habitat-challenge /root/midlevel-reps/habitat-challenge

WORKDIR /root/midlevel-reps
RUN pip install -r requirements.txt
RUN ln -s habitat-api/data .


###############################
#  set up baselines
###############################
WORKDIR /root
RUN apt-get update && apt-get install -y cmake libopenmpi-dev python3-dev zlib1g-dev
RUN git clone https://github.com/openai/baselines.git; cd baselines; pip install -e .


###############################
#  set up habitat-challenge
###############################
ADD submission.sh /submission.sh
ADD eval_runs /mnt/eval_runs
RUN ln -s /root/midlevel-reps/habitat-api/data/habitat-challenge-data /

######################################
# install tnt
######################################

WORKDIR /root/midlevel-reps/tnt
RUN  pip install -e .

RUN apt-get install -y screen

WORKDIR /root/midlevel-reps
