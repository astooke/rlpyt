import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'torch',
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # unmandatory dependencies of the package itself
    'atari_py', 'opencv-python', 'psutil', 'pyprind', 'gym',
]

setuptools.setup(
    name='rlpyt',
    version='0.1.1dev',
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },

)
