from setuptools import setup, find_packages

setup(
    name='twitter_profile_predictor',
    version='0.1.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'nltk==3.7',
        'pandas==2.1.4',
        'asttokens==2.0.5',
        'openpyxl==3.0.10'
        # Add other dependencies as needed see in the original conda environment wich version are used !!!!
    ],
    author='Tim Faverjon',
    author_email='tim.faverjon@sciencespo.fr',
    description='A python module that extract information (language, profession...) from X[ex-Twitter] profile bios',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/TimFaverjon/twitter_profile_predictor',
    package_data={
        'twitter_profile_predictor': ['data/*.xlsx'],
    }

)