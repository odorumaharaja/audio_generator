'''
audio generator for Keras

reference
https://github.com/alibugra/audio-data-augmentation
https://github.com/sid0710/audio_data_augmentation
https://www.kaggle.com/CVxTz/audio-data-augmentation
'''

import librosa
import numpy as np

class AudioGenerator(object):
    def __init__(self, X_train, y_train, batch_size=32, shuffle=True, seed=12345):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(X_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            i = 0
            while i < self.sample_num:
                if i + self.batch_size < self.sample_num:
                    X, y = self.__data_generation(indexes[i:i + self.batch_size])
                else:
                    X, y = self.__data_generation(indexes[i:])
                
                i += self.batch_size
                yield X, y
       
    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        X = self.X_train[batch_ids]
        y = self.y_train[batch_ids]
        
        # audio argumentation
        
        return X, y

 
if __name__ == '__main__':


    # Read and show cat sound
    data = librosa.core.load("data/cat.wav", sr=16000)[0]
    frames = librosa.util.frame(data, frame_length=2048, hop_length=512)
    labels = np.zeros(frames.shape[0],)
    print(frames.shape, labels.shape)
    aa = AudioGenerator(frames, labels, batch_size=200)
    for X, y in aa():
        print(X.shape, y.shape)


