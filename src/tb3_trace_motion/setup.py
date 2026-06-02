import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'tb3_trace_motion'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Odom-based TurtleBot3 trace motion paths with a Qt editor.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'motion_executor = tb3_trace_motion.motion_executor_node:main',
            'motion_ui = tb3_trace_motion.motion_ui_node:main',
        ],
    },
)
