from setuptools import setup

setup(
    name="ai_movie_posters",
    version="0.1.0",
    description="Predicting the genre of a movie by analyzing its poster",
    url="https://github.com/acanda/ai-movie-posters",
    author="Philip Graf",
    license="Apache Software License 2.0",
    install_requires=[
        "pandas",
        "numpy",
        "Pillow",
        "keras",
        "h5py",
        "pydot-ng",
        "graphviz",
        # tensorflow-gpu is much faster than tensorflow but you need a GPU that supports CUDA
        # "tensorflow",
        "tensorflow-gpu"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha"
    ]
)
