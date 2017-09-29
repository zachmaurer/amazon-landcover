# Amazon Rainforest Land Cover Prediction
Dataset from Planet Labs via Kaggle

Objectives:

* Given a 3-channel JPG satellite image tile, classify the image as containing one or more of 17 labels.

## Summary of Findings

* Random Forest classifiers with remarkably naive features are incredibly competitive baselines
* More work could be done on creating a semantically interpretable feature set (i.e. blob detection indicator variables) and feeding that to a decision tree
* Recursive neural networks, like the U-Net we implemented, did not perform particularly well. This was a surpise since recursive structures have performed well on image segmentation tasks
* Simple CNN networks performed similarly to a U-Net.
* U-Net sampling, concatenation operations were believed to slow the runtime down siginificantly.
* Creating a new loss function that incorporated a Jaccard similarity metric did little to change the results
* Transfer learning, from GoogLe Net or ResNet worked best. Extremely small LR and many epochs (> 50) produced the best results. 
* PyTorch is an incredible tool for prototyping resarch networks.

Citations are in the references section of the poster.

## Poster

![Poster](https://github.com/zachmaurer/amazon-landcover/raw/master/assets/231n-poster.jpg)


