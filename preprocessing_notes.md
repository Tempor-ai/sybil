# Notes on preprocessing used in the following two notebooks
https://www.kaggle.com/code/koheimuramatsu/change-detection-forecasting-in-smart-home <br>
https://www.kaggle.com/code/piergiacomofonseca/smart-home-iot-eda-arimas-lstm-and-more

Sanity checks:
-   Make sure column have the expected types (float64 not object, pd.DateTime)
-   Does the dataset have the expected granularity? ex. Dataset with 1h granularity spanning 24h should contain 24 rows.
-   Remove duplicate columns

Missing Values
    - First notebook drops contains only one missing row which is dropped.

Feature engineering
-   Create day of week, hour of day, morning/afternoon/night columns


Feature Importance Analysis
- They have used linear correlation but one needs to be careful with that (confounders, non-linear relationships, etc.).We probably should use a more robust feature importance analysis method (RFE, Permutation Feature Importance, etc. Need to find a robust study comparing these.)
- 