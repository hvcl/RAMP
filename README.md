# RAMP
This is an official implementation of "RAMP: Response-Aware Multi-task Learning with Contrastive Regularization for Cancer Drug Response Prediction" by [Kanggeun Lee](https://scholar.google.com/citations?hl=ko&user=OvRs1iwAAAAJ), Dongbin Cho, Jinho Jang, Hyoung-oh Jeong, Jiwon Seo, Won-Ki Jeong and Semin Lee.


## **Environment Setup** ##
The environment settings for RA-NS are described in [here](src/embed/README.md).

Building docker images to test 10-fold nested cross validation using extracted embedding vectors.
```
cd docker/tensorflow
./build.sh 
or
cd docker/pytorch
./build.sh 
```

# Quick test
```
cd src/response_prediction/scripts
./demo_tf.sh or ./demo_torch.sh # Choose your environment either tensorflow or pytorch. All experiments in the manuscript were done in Tensorflow.
```

#
