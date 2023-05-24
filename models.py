import os, cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.keras.applications import InceptionV3, ResNet50, Xception
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import image_dataset_from_directory
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

class TB():
    def __init__(self):
        self.shape = (224, 224, 3)
        self.classes = 219
        self.batch_size = 16

    def load_data(self, dir):
        train = image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.shape[0], self.shape[1]),
            interpolation="bicubic",
            batch_size=self.batch_size)

        valid = image_dataset_from_directory(
            dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.shape[0], self.shape[1]),
            interpolation="bicubic",
            batch_size=self.batch_size)

        return train, valid

    def inceptionv3(self):
        inc = InceptionV3(include_top=False, weights="imagenet",
               input_shape=(self.shape[0], self.shape[1], self.shape[2]))
        for i in inc.layers:
            i.trainable = False
        x = LeakyReLU(alpha=0.2)(inc.output)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(self.classes, activation="softmax")(x)
        model = Model(inputs=inc.input, outputs=x)
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=0.0005)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        

        return model

    def resnet50(self):
        res = ResNet50(include_top=False, weights="imagenet",
               input_shape=(1, self.shape[0], self.shape[1], self.shape[2]))
        for i in res.layers:
            i.trainable = False
        x = LeakyReLU(alpha=0.2)(res.output)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(self.classes, activation="softmax")(x)
        model = Model(inputs=res.input, outputs=x)
        opt = SGD(learning_rate=0.001, momentum=0.9, decay=0.0005)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def xception(self):
        xcep = Xception(include_top=False, weights="imagenet",
               input_shape=(self.shape[0], self.shape[1], self.shape[2]))
        for i in xcep.layers:
            i.trainable = False
        x = LeakyReLU(alpha=0.2)(xcep.output)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(self.classes, activation="softmax")(x)
        model = Model(inputs=xcep.input, outputs=x)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def train(self, epochs):
        train, valid = self.load_data(dir="all_aug")
        # model = self.inceptionv3()
        model = self.resnet50()
        # model = self.xception()
        # model.fit(x=train, y=train_label, validation_split=0.3, epochs=epochs, batch_size=batch_size)
        # checkpointer = ModelCheckpoint(filepath="model_osme_inceptionv3.best_loss.hdf5", verbose=1, save_best_only=True)
        checkpointer = ModelCheckpoint(filepath="model_osme_resnet50.best_loss.hdf5", verbose=1, save_best_only=True)
        # checkpointer = ModelCheckpoint(filepath="model_osme_xception.best_loss.hdf5", verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                                    patience=5, min_lr=0.0000001)
        model.fit(train, epochs=epochs, validation_data=valid, callbacks=[reduce_lr, checkpointer])
        score = model.evaluate(train)
        print("\nTrain Loss:", score[0])
        print("\nTrain Acc:", score[1])
        # model.save("inceptionv3.h5")
        # model.save("resnet50.h5")
        # model.save("xception.h5")

if __name__ == "__main__":
    tb = TB()
    tb.train(epochs=100)