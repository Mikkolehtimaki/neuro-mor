from setuptools import setup

setup(
    name='model_reduction',
    version='1.0',
    description='Defining and reducing neuroscience ODE models in python',
    license='MIT',
    author='Mikko Lehtimäki',
    author_email='mikko.lehtimaki@tuni.fi',
    packages=['model_reduction', 'model_reduction.models'],
    install_requires=[
        'numpy>=1.14.3',
        'matplotlib>=2.2.2',
        'jupyter'
    ],
    scripts=[]
)
