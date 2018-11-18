from setuptools import setup
from setuptools import find_packages

setup(
    name='ftdd',
    version='1.0',
    description='Fraudulent Taxi Driver Detection',
    author='Javy Wang',
    author_email='javyw94@gmail.com',
    maintainer=['Javy Wang',
                'Tianle Ni'
                ],
    maintainer_email=['javyw94@gmail.com', ''],
    url='',
    download_url='',
    license='MIT',
    install_requires=['numpy',
                      'networkx',
                      'pandas'],
    packages=find_packages()
)

