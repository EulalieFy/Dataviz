# Mini Vast Challenge 2018

This project was made to answer to the data visualization contest : [Mini Vast Challenge](http://www.vacommunity.org/VAST+Challenge+2018)

Kasios is accused of damaging the bird's living area of the Lekagul Nature Reserve. Our goal is to use Data Visualization, Signal Processing (for birds sounds analysis) and machine learning to determine if they are guilty of fleeing the birds from the reserves. The type of birds we are interested in is Rose-crested Blue Pipit.

Kasios pretends that they have recently recorded 15 blue-pipits, for each of this recordings we are given the sound file and the localization (lat and lng). Our projects aims at finding if they are really blue pipits.

- First step in [this notebook](https://github.com/EulalieFy/Mini-Vast-Challenge-2018/blob/master/Birds%20Location.ipynb): plot each bird localisation and test localization on a map to see whether it was logical.

- Second step in [this notebook](https://github.com/EulalieFy/Mini-Vast-Challenge-2018/blob/master/Sound%20Visualizations.ipynb): visualize birds sounds ( waveform and spectogram ) to see if we can find resemblances. We will focus on blue pipits to find out which visual features caracterizes their sound (call / song).

- Third step in [this notebook](https://github.com/EulalieFy/Mini-Vast-Challenge-2018/blob/master/Features%20extraction.ipynb):  compute features from birds sounds, that will be used to make predictions

- Fourth step in [this notebook](https://github.com/EulalieFy/Mini-Vast-Challenge-2018/blob/master/Classification%20and%20Conclusion.ipynb): use machine learning algorithm to predict which bird is behind each recordings.

- Fifth step in [this folder](https://github.com/EulalieFy/Mini-Vast-Challenge-2018/tree/master/Dashboard): building a dashboard to enabling the user to draw conclusion from localization, sound visualizations and predictions for each birds.

To run the Dashboard on your laptop, download this folder and run this command on your terminal
```
bokeh serve --show bokeh_final.py
```
