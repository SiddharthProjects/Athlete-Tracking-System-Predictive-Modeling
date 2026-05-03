from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="employee-training-ml",
    version="1.0.0",
    author="Priti Ranjan Samal",
    author_email="",
    description="Comprehensive ML pipeline for intelligent employee training and resource management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SiddharthProjects/employee-training-ml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    keywords="machine-learning xgboost employee-training course-recommendation",
    project_urls={
        "Bug Reports": "https://github.com/SiddharthProjects/employee-training-ml/issues",
        "Source": "https://github.com/SiddharthProjects/employee-training-ml",
        "Documentation": "https://github.com/SiddharthProjects/employee-training-ml/blob/main/Readme.md",
    },
)
