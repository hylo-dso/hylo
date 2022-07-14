import setuptools

setuptools.setup(
    name="hylo",
    version="0.0.1",
    author="anonymous",
    author_email="hylo.dso@outlook.com",
    description="Hybrid Low-rank Distributed Second-order Optimizer",
    long_description=open('README.md').read(),
    url="https://github.com/hylo-dso/hylo-src",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch >= 1.7"
    ],
)
