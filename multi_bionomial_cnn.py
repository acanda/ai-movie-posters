import pandas as pd
import numpy as np
from pathlib import Path
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard
from PIL import Image
import time


input_path = Path("input")

poster_width = 96  # 182 / 3.7916
poster_height = 128  # 268 / 4.1875
poster_channels = 'RGB'

epochs = 25
batch_size = 256


def load_genres(path):
    csv = path / 'MovieGenre.csv'
    df = pd.read_csv(csv, encoding="ISO-8859-1", usecols=['Genre'], keep_default_na=False)
    genres = set()
    for gs in df.Genre.str.split('|').values:
        genres.update(g for g in gs if len(g) > 0)
    genres = sorted(genres)
    print(f"{len(genres)} genres:", ', '.join(genres))
    return genres


def create_cnn(height, width, channels):
    cnn = Sequential([
        Conv2D(filters=32, kernel_size=(9, 9), activation="relu", input_shape=(height, width, channels)),
        Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(loss=binary_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    return cnn


def poster_file(path, imdb_id):
    return path / 'SampleMoviePosters' / f'{imdb_id}.jpg'


def load_poster(path, imdb_id):
    with Image.open(poster_file(path, imdb_id)) as poster:
        poster = poster.resize((poster_width, poster_height), resample=Image.LANCZOS)
        poster = poster.convert(poster_channels)
        return np.asarray(poster) / 255


def load_data(path, genre):
    csv = path / 'MovieGenre.csv'
    df = pd.read_csv(csv, encoding="ISO-8859-1", usecols=['imdbId', 'Genre'], keep_default_na=False)
    df = df[df.imdbId.map(lambda imdb_id: poster_file(path, imdb_id).is_file())]
    df = df[df.Genre.map(lambda g: len(g) > 0)]
    df = df.sample(frac=1).reset_index(drop=True)
    x = np.array(df.imdbId.map(lambda imdb_id: load_poster(path, imdb_id)).values)
    y = np.array(df.Genre.str.split('|').map(lambda gs: genre in gs).values)
    separator = len(x) * 3 // 4
    return x[:separator], y[:separator], x[separator:], y[separator:]


movie_genres = load_genres(input_path)
for movie_genre in movie_genres:
    model = create_cnn(poster_height, poster_width, len(poster_channels))
    x_train, y_train, x_validation, y_validation = load_data(input_path, movie_genre)

    tensor_board = TensorBoard(log_dir=f"./logs/{time.strftime('%Y-%m-%d %H.%M.%S')}", write_images=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_validation, y_validation),
                        callbacks=[tensor_board])

