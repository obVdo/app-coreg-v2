FROM brainlife/mne-freesurfer:7.3.2-1.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*
