from setuptools import setup

package_name = 'yolo_v8_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoon',
    maintainer_email='yoon@example.com',
    description='YOLOv8 ROS2 node for remote PC inference',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_v8_ros.yolo_node:main',
        ],
    },
)
