from setuptools import find_packages, setup

setup(
    name="comlrl",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    packages=find_packages(),
    python_requires=">=3.8",
)
