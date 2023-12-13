from setuptools import setup, find_packages

setup(
    name='twitter-profile-predictor',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'nltk==3.7',
        'pandas==1.5.3',
        'asttokens==2.0.5'
        # Add other dependencies as needed see in the original conda environment wich version are used !!!!
    ],
    entry_points={
        'console_scripts': [
            'my_script = src.profile_predictor_twitter.bios_analyzer',
        ],
    },
    author='Tim Faverjon',
    author_email='tim.faverjon@sciencespo.fr',
    description='A python module that extract information (language, profession...) from X[ex-Twitter] profile bios',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TimFaverjon/twitter_profile_predictor',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    package_data={
        'profile_predictor_twitter': ['data/*.xlsx'],
    }

)