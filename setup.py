import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name='background_generation',
    version='0.1.0',
    python_requires='>=3.7.0',
    packages=['background_generation'],
    include_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)