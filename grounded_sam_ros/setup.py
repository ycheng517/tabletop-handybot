from setuptools import setup

package_name = 'grounded_sam_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
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
    description='Example of using Grounded-SAM in ROS 2.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'grounded_sam_node = grounded_sam_ros.grounded_sam_node:main',
            'audio_prompt_node = grounded_sam_ros.audio_prompt_node:main',
        ],
    },
)
