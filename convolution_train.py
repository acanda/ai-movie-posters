import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard


input_path = "input"
poster_width = 48  # 182 / 3.7916
poster_height = 64  # 268 / 4.1875


def poster_file(movie):
    return input_path + "/SampleMoviePosters/" + str(movie[0]) + ".jpg"


def has_poster(movie):
    return Path(poster_file(movie)).is_file()


def has_genre(movie):
    return len(movie[1]) > 0


def movie_poster(movie):
    poster = Image.open(poster_file(movie))
    poster = poster.resize((poster_width, poster_height), resample=Image.LANCZOS)
    poster = poster.convert('RGB')
    return np.asarray(poster) / 255


def unique(series_of_lists):
    seen = set()
    return [e for lst in series_of_lists
            for e in lst
            if not (e in seen or seen.add(e))]


def bitmap(lst, uniques):
    bmp = []
    for u in range(0, len(uniques)):
        if uniques[u] in lst:
            bmp.append(1.0)
        else:
            bmp.append(0.0)
    return bmp


def encode(series, uniques):
    return [bitmap(lst, uniques) for lst in series]


def load_data():
    movies = pd.read_csv(input_path + "/MovieGenre.csv", encoding="ISO-8859-1", usecols=['imdbId', 'Genre'], keep_default_na=False)
    movies = movies[movies.apply(lambda d: has_genre(d) and has_poster(d), axis=1)]
    movies = movies.sample(frac=1).reset_index(drop=True)
    posters = list(map(movie_poster, movies.values))
    genres = movies.Genre.str.split("|")
    unique_genres = unique(genres)
    x = np.array(posters)
    y = np.array(encode(genres.values, unique_genres))
    return x, y, unique_genres


x_data, y_data, genres = load_data()
separator = len(x_data) * 3 // 4
x_train = x_data[0:separator]
y_train = y_data[0:separator]
x_test = x_data[separator:len(x_data)]
y_test = y_data[separator:len(y_data)]

model = Sequential([
    Conv2D(filters=128, kernel_size=(4, 4), activation="relu", input_shape=(poster_height, poster_width, 3)),
    Conv2D(filters=64, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(genres), activation='softmax')
])

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

tensorboard = TensorBoard(write_images=True)

epochs = 5
batch_size = 256
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard])

model.save(f"models/conv-model-e{epochs}-b{batch_size}.h5")
model.save_weights(f"models/conv-weights-e{epochs}-b{batch_size}.h5")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\nInput (from training data):", y_data[0:3])
print("Prediction:", model.predict(x_data[0:3]))

print("\nInput (from test data):", y_data[-4:-1])
print("Prediction:", model.predict(x_data[-4:-1]))

print("\nGenres:", genres)

keras.utils.plot_model(model, to_file=f"models/convolution.png", show_shapes=True)
