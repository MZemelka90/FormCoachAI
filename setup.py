from setuptools import setup

setup(
    name='formcoachai',
    version='0.1.0',
    description='FormCoachAI: A computer vision system for analyzing human movement patterns',
    author='Maximilian Fuchs',
    author_email='maximilian.fuchs@fau.de',
    url='https://github.com/MZemelka90/FormCoachAI',
    packages=['formcoachai'],
    install_requires=[
        'numpy',
        'opencv-python',
        'mediapipe'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
