General Task
============
- For a potentially big number of URLs a parallelized python implementation is provided
- The implementation itself is rather simplistic, since a lot depends on specifics
and restrictions imposed by production environment and detailed problem setting
```
# example usage on linux
$python3.4 general.py -f ./links -d ./img
```


Data Scientist Task
===================
- After visualization of different features against each other and the dependant variable 
(total count of visitors) several findings were made:
    - provided data is a non-stationary time series subjected to seasonality and a general 
    increasing trend (with a sharp locally decreasing trend towards the end of 2012)
    - there is a noticeable correlation between apparent temperature and both registered and 
    casual users, which seem to contribute equally to both groups
    - naturally, there are more users during good weather days as compared to poor weather ones
- It seems only wise to use time series analysis (TSA) modelling approach and decompose the 
data into stationary, trend and seasonality and perform analysis and modelling from that point 
(e.g. using ARIMA model). However, it would require a bit of learning from my end and the time 
limit doesn't permit that. Consequently, it was decided to use a simple ridge regression model 
for provided non-linear data set:
    - some features were dropped (i.e. humidity, temperature, etc.) since they were implicitly 
    present and combined in apparent temperature feature
    - no dimensionality reduction methods were tested for the sake of simplicity
    - categorical and most time related features were transformed using one-hot-encoding approach 
    due to an explicit bias and misleading information (e.g. a distance measure of 1h and 23h would 
    suggest that the values are quite far apart, however they are merely 2h away)
    -  predicted value wasn't scaled back, however in a real world usage scenario such operation 
    is obviously required
```
# example usage on linux
$python3.4 data-scientist.py -f ./data/hour.csv
```

Code dependencies
=================
- python 3.4.3
- requests 2.11.1
- pandas 0.18.1
- numpy 1.11.1
- matplotlib 1.5.2
- scikit-learn 0.17.1