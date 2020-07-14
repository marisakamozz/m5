#!/bin/sh
mkdir data
mkdir input
mkdir logs
mkdir submissions
kaggle competitions download -c m5-forecasting-accuracy
kaggle competitions download -c m5-forecasting-uncertainty
kaggle datasets download gsnehaa21/federal-holidays-usa-19662020
mkdir input/m5-forecasting-accuracy
mkdir input/m5-forecasting-uncertainty
mkdir input/holiday
unzip m5-forecasting-accuracy.zip -d input/m5-forecasting-accuracy
unzip m5-forecasting-uncertainty.zip -d input/m5-forecasting-uncertainty
unzip federal-holidays-usa-19662020.zip -d input/holiday
