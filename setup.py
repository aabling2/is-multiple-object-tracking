from setuptools import setup, find_packages


def get_requirements(req_file="requirements.txt"):
    lines = [line.strip() for line in open(req_file)]
    return [line for line in lines if line]


setup(
    name='is_multiple_object_tracker',
    version='0.1',
    description='',
    url='aabling2/is-multiple-object-tracking',
    author='labvisio',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'is-multiple-object-tracker=is_mot_bytetrack.main:main'
        ],
    },
    zip_safe=False,
    install_requires=get_requirements(req_file="requirements.txt"),
)