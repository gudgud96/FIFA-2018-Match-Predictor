## FIFA World Cup 2018 Predictor

This is the code site for predicting FIFA World Cup results using ML techniques - linear regression and neural network. 

For running preprocess and linear regressor / neural network respectively, use the following command(s):
```
python fifa-results-preprocess.py
python fifa-ml.py
python fifa-ml-nn.py
```

Prediction results as below:

Top 4 by Linear Regressor: Brazil, Spain, Argentina, Germany

Top 4 by neural network: Brazil, Spain, Germany, Portugal

Please find both result .csv files for prediction outcomes, including win, lose and draw probabilities.

For details, kindly refer to [this blogpost](http://gudgud96.github.io/projects/fifa-18.html) on my personal tech blog.

Summary: It seems like football predictions couldn't be accurately predicted merely using ML techniques. Training and testing accuracy could only hit around 50%, which either means that (i) a decent model is not yet found, or (ii) football match results are unpredictable.