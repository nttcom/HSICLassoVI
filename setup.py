import setuptools

with open('README.md', 'r') as fh:
    README = fh.read()

setuptools.setup(
    name='HSICLassoVI',
    version='1.0',
    author='K.Koyama, K.Kiritoshi, T.Okawachi, T.Izumitani',
    description='"Effective Nonlinear Feature Selection Method based on HSIC Lasso and with Variational Inference" Python Package',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=[
        'joblib',
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn'
    ],
    url='https://github.com/koyama-com/HSICLassoVI',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)