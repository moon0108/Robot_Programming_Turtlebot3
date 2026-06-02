import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'tb3_delivery_core'


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
    description='Core TurtleBot3 delivery mission controller.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'delivery_master = tb3_delivery_core.delivery_master_node:main',
            'qt_order = tb3_delivery_core.qt_order_node:main',
            'aruco_process_manager = tb3_delivery_core.aruco_process_manager_node:main',
        ],
    },
)
