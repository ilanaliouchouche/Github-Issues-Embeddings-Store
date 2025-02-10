from setuptools import setup, find_packages

setup(
    name='GitHub-Issues-Embeddings-Store',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Ilan Aliouchouche',
    install_requires=[
        'numpy',
        'pandas',
        'transformers',
        'accelerate',
        'matplotlib',
        'tqdm',
        'datasets',
        'tokenizers',
        'sentence-transformers',
        'xformers',
        'tensorboard',
        'optuna',
        'bs4',
        'seaborn',
        'torch'
    ],
    python_requires='>=3.11',
)