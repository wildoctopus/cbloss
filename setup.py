from setuptools import setup, find_packages

setup(
    name='cbloss',
    version='0.1.0',
    packages=find_packages(),
    license='MIT',
    description='PyTorch implementations of two popular loss functions for imbalanced classification problems: Class Balanced Loss and Focal Loss.',
    author='Alok Pandey',
    url='https://github.com/yourusername/classbalancedloss-pypy',
    keywords=['pytorch', 'loss', 'class-balanced', 'classification-loss', 'focal-loss', 'FocalLoss', 'cb_loss', 'cbloss'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.19.2'
    ],
)
