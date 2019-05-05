# !/bin/bash

ENV=DL

echo Creating environment
conda env remove --name $ENV
conda create --yes --name $ENV

echo Setup complete