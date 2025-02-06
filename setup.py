# setup.py
from setuptools import setup, find_packages

setup(
    name='HMSMC_stock_dynamics',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'torch',
        'pandas',
        'scikit-learn',
        'pywavelets',
        'yfinance',
        'tiingo',
        'matplotlib',
        'seaborn',
        'tqdm',
        'jupyter',
        'scipy',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here if needed
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)