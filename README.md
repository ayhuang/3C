# 3C - classic composer classifier
## A transformer based classifier of classic music composers 
### Date set
Samples are classic music midi files gathered from open sources. The majority of works by top composers, such as Bach, Mozart and Beethoven is collected. Each midi file may be an entire piece or movements of large pieces. Each file is parsed with the [MIT music21 library](https://web.mit.edu/music21/) into stream, then each measure of the stream is encoded according to the scheme developed by [musicautobot](https://github.com/bearpelican/musicautobot) into a 2-d numpy array of note pitch and duration. In training,  a hyper parameter - number of measures per sample is used to generate samples based on continuous measures to feed into the network. About half a million of measures are encoded, of which around 20 to 25% are randomly selected to be used for training and validation.

### transformer network
As this is a simple classifying problem, only the encoder part of the transformer network is used, with a dense layer on top to produce the category (composer) predictions. The top composers with voluminous works have individual labels, the less prolific ones are grouped as "other" for a balanced distribution among the categories. 
With about 15000 samples, a validation accuracy over 60% is obtained, while the training accuracy is close to 80%, there is certain degree of overfitting, but nontheless, given the nature of the problem, it is hard to imagine a human expert to reach very high accuracy with such a vast number of classic music pieces. Of course, one could expect a deep learning model to do better than human in the fun exercise of identifying a classic composer. With more tuning and different ways of encoding, one may just achieve that.

