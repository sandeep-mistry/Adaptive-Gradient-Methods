# !/bin/bash

ENV=DL1

echo Creating environment
conda env remove --name $ENV
conda create --yes --name $ENV

echo Setup complete
