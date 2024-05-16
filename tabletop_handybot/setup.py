import os
from glob import glob
from setuptools import setup

package_name = 'tabletop_handybot'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages',
                 ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml']),
                (os.path.join('share', package_name, 'launch'),
                 glob(os.path.join('launch', '*launch.[pxy][yma]*')))],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Yifei Cheng',
    author_email='ycheng517@gmail.com',
    maintainer='Yifei Cheng',
    maintainer_email='ycheng517@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='A handy robot arm for performing tabletop tasks.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tabletop_handybot_node = tabletop_handybot.tabletop_handybot_node:main',
            'audio_prompt_node = tabletop_handybot.audio_prompt_node:main',
        ],
    },
)
