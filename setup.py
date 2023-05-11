from setuptools import setup, find_packages
import io
import os
import re



def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "cb_loss", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


_DEV_REQUIREMENTS = [
    "black==21.7b0",
    "flake8==3.9.2",
    "isort==5.9.2",
    "click==8.0.4",
    "importlib-metadata>=1.1.0,<4.3;python_version<'3.8'",
]

extras = {"tests": _DEV_REQUIREMENTS, "dev": _DEV_REQUIREMENTS}

setup(
    name='cbloss',
    version=get_version(),
    license='MIT',
    description='PyTorch implementations of two popular loss functions for imbalanced classification problems: Class Balanced Loss and Focal Loss.',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author='Alok Pandey',
    url='https://github.com/wildoctopus/cbloss',
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    install_requires=get_requirements(),
    extras_require=extras,
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=['pytorch', 'loss', 'class-balanced', 'classification-loss', 'focal-loss', 'FocalLoss', 'cb_loss', 'cbloss'],
    
)
