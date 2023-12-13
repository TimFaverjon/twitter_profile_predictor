from setuptools import setup, find_packages

setup(
    name='my-package',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'nltk==3.6.3',
        'pandas==1.3.3',
        'openpyxl==3.0.14',
        # Add other dependencies as needed see in the original conda environment wich version are used !!!!
    ],
    entry_points={
        'console_scripts': [
            'my_script = my_package.module1:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='Description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my-package',
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