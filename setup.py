from setuptools import setup, find_packages

setup(
    name="joy",
    version="0.1.0",
    packages=find_packages(),
    description="Formatter for aligning assignment expressions",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Joseph S. Boyle",
    author_email="joespartacusboyle@gmail.com",
    url="https://github.com/JosephSBoyle/joy_formatter",
    install_requires=[],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
)
