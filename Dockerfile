FROM continuumio/anaconda3
COPY environment.yml .
COPY . .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/provim/bin:$PATH
RUN /bin/bash -c "source activate provim"