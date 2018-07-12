# Health_prediction

***The 2017 code and results are uploaded (in old_code and old_data).***

This project is that predict next health data using electronic medica record
Discription can be found in [health_prediction_poster.pdf](health_prediction_poster.pdf) and [health_prediction_abstract.pdf](health_prediction_abstract.pdf).


## Project Overview
This project describes the development of a model for predicting next health feature values of electronic medical record (EMR). Model is implemented ensemble based. To do this, I used the MIMIC-III database which contains details records from ~60,000 ICU admissions for ~40,000 patients over a period of 10 years. Using these records, all of patient records recorded by time were preprocessed for prediction of next record. 


### Data
This project uses the MIMIC-III database :
https://mimic.physionet.org/

To properly run the project from the included files, a local postgres SQL server must be installed and the MIMIC-III database must be set up as described in https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic.

The data set for prediction was extracted from the database as defined in [feature_preprocessing.ipynb](data/feature_preprocessing.ipynb).
