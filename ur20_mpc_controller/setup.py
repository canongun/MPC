from setuptools import setup, find_packages

setup(
    name="ur20_mpc_controller",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'rospy',
        'scipy'
    ],
)