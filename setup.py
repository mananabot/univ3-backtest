from setuptools import setup, find_packages


setup(
    name="univ3-backtest",
    description="A simple backtesting framework for Uniswap v3 LP strategies.",
    author="mananabot",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'pandas', 'matplotlib'
    ]
)
