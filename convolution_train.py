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

input_path = Path("input")
model_path = Path("models")

poster_width = 48  # 182 / 3.7916
poster_height = 64  # 268 / 4.1875
poster_channels = 3  # RGB

epochs = 2
batch_size = 256


def poster_file(movie):
    return input_path / "SampleMoviePosters" / f"{movie[0]}.jpg"


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


def load_data(path):
    csv = path / "MovieGenre.csv"
    movies = pd.read_csv(csv, encoding="ISO-8859-1", usecols=['imdbId', 'Genre'], keep_default_na=False)
    movies = movies[movies.apply(lambda d: has_genre(d) and has_poster(d), axis=1)]
    movies = movies.sample(frac=1).reset_index(drop=True)
    posters = list(map(movie_poster, movies.values))
    genres = movies.Genre.str.split("|")
    unique_genres = unique(genres)
    x = np.array(posters)
    y = np.array(encode(genres.values, unique_genres))
    return x, y, unique_genres


def create_cnn(height, width, channels, genres):
    cnn = Sequential([
        Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(height, width, channels)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(genres), activation='softmax')
    ])
    cnn.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    return cnn


x_data, y_data, movie_genres = load_data(input_path)
separator = len(x_data) * 3 // 4
x_train = x_data[0:separator]
y_train = y_data[0:separator]
x_validation = x_data[separator:len(x_data)]
y_validation = y_data[separator:len(y_data)]

model = create_cnn(poster_height, poster_width, poster_channels, movie_genres)
model.summary()

tensorboard = TensorBoard(write_images=True)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_validation, y_validation),
                    callbacks=[tensorboard])

model_path.mkdir(parents=True, exist_ok=True)
model.save(model_path / f"conv-model-e{epochs}-b{batch_size}.h5")
model.save_weights(model_path/f"conv-weights-e{epochs}-b{batch_size}.h5")

score = model.evaluate(x_validation, y_validation, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\nInput (from training data):", y_data[0:3])
print("Prediction:", model.predict(x_data[0:3]))

print("\nInput (from validation data):", y_data[-4:-1])
print("Prediction:", model.predict(x_data[-4:-1]))

print("\nGenres:", movie_genres)

keras.utils.plot_model(model, to_file=str(model_path / "convolution.png"), show_shapes=True)
