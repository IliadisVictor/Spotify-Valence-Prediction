# Spotify-Valence-Prediction

## Context
Psychologists use the term valence , to describe whether something is likely to make someone feel happy or sad , something like the electron sense of the word . Spotify uses a metric called valence to indicate the ability of a song to affect the listener in a happy or a negative way . The metric was originally created by a company called Echo Nest that was acquired by Spotify in 2014.


## Task

The task is to study which of the features are of statistical importance , and have predictive power in term of predicting valence , and use them to find the best non-connectivist method for predicting the target value , and the best neural network method . 

## Data Extraction - ETL file
Initially we used a ready [Zenodo](https://zenodo.org/record/4778563#.YgAF4bpBy3A) dataset , to get the charts for 9 different regions from 17' to 20' . Then we used [Spotify's  API](https://developer.spotify.com/)  to get for each track the available features and analysis metrics . You can find the complete process in [ETL.ipynb](https://github.com/IliadisVictor/Spotify-Valence-Prediction/blob/main/ETL.ipynb) , and the complete dataset for ease in the TrackData Folder.

## Features Selection
Filter Methods tried :
* Pearson's Correlation
* Spearman's Correlation

In the end we didn't move forward with a filter method but we did use Spearman's correlation to remove highly inter correlated features . 

Multiple Linear Regression :
* Backward stepwise selection
* Forward stepwise selection

These methods had the same results and we ended up using 37 features .More info on the approaches for finding feature significance i mentioned can be found [here](https://quantifyinghealth.com/stepwise-selection/)

## Non Connectivist Approaches
* LightGBM - MAE 0.0953
* CatBoost - MAE 0.0966
* XGB - MAE 0.0984
* Random Forest - MAE 0.1127
* Linear Regression - MAE 0.1228
* Decision Tree - MAE 0.1439
* Voting Ensemble (XGB and Light) - MAE 0.0942
  
**Best Result** Stacked Ensemble XGB and Light GBM to a final estimator of Linear Regression MAE 0.0935

### Hyper Parameter tuning
In the notebook you can find for each of the regressors used a section for tuning it's hyper parameters that were deemed most important for affecting the performance on your dataset. In some cases we tried a larger search space with a randomized search , and in other's a smaller search space with an exhaustive grid search .  

### Important Features of Stacked Ensemble

For XGB:
* Pitch 9
* tempo
* Pitch 6
* Pitch 12

For Light GBM:
* Pitch 1
* Pitch 12
* Timbre 7
* Total Segments

for Linear Regression

* Timbre 1
* Total Segments
* Pitch 12
* loudness

## Neural Networks

First common layer in all models was a normalization layer, we run 4 models in Tensorflow with differentiations  in the layers , the optimal results came from :
* 3 relu activated layers , and an Adam optimizer  

