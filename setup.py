# Author: Adrien Corenflos

"""Install."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='bfs-project-code',
    version='0.1',
    description='Maximum likelihood linearization state.',
    author='Adrien Corenflos',
    author_email='adrien.corenflos@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
