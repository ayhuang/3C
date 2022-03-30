# 3C - classic composer classifier
## A transformer based classifier of classic music composers 
### Date set
Samples are classic music midi files gathered from open sources. The majority of works by top composers, such as Bach, Mozart and Beethoven is collected. Each midi file may be an entire piece or movements of large pieces. Each file is parsed with music21 library from MIT into stream, then each measure of the stream is encoded according to the scheme developed by musicAutocoder into a 2-d numpy array of note pitch and duration. In training a hyper parameter, number of measures per sample is used to generate samples based on continuous measures to feed into the network. A close 
