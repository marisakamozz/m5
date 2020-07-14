#!/bin/sh
cd src
# python train.py trainer=agg_cv
# python train.py trainer=each_cv
python train.py trainer=agg_submit
python train.py trainer=each_submit
# 
# Temporal submission DataFrames will be saved at /data/submissions .
# Filename format is 
# 
#   [agg/each]_submit-[acc/unc]-[%Y%m%d]-[%H%M%S].joblib
# 
# for example:
# 
#   agg_submit-acc-20200626-013248.joblib
#   agg_submit-unc-20200626-013248.joblib
#   each_submit-acc-20200628-144217.joblib
#   each_submit-unc-20200628-144217.joblib
# 
# To create submission files from these temporal submission DataFrames,
# please run following command.
# 
#   python makesub.py [agg timestamp] [each timestamp]
# 
# for example:
# 
#   python makesub.py 20200626-013248 20200628-144217
# 
# Then, submission files will be created at /submissions .
# Filename format is
# 
#   [acc/unc]-[%Y%m%d]-[%H%M%S].csv
# 
# for example:
# 
#   acc-20200630-001300.csv
#   unc-20200630-001300.csv
# 