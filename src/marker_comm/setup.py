import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'marker_comm'


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        files = [os.path.join(path, filename) for filename in filenames]
        if files:
            paths.append((os.path.join('share', package_name, path), files))
    return paths


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
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ] + package_files('models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='min',
    maintainer_email='jmsm1378@kw.ac.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'checkerboard_capture = marker_comm.checkerboard_capture_node:main',
            'calibrate_camera = marker_comm.calibrate_camera:main',
            'generate_aruco_markers = marker_comm.generate_aruco_markers:main',
            'aruco_pose = marker_comm.aruco_pose_node:main',
            'aruco_align_controller = marker_comm.aruco_align_controller_node:main',
        ],
    },
)
