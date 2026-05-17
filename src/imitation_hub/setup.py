from setuptools import find_packages, setup

package_name = 'imitation_hub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irfan',
    maintainer_email='irfanahmed7457@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_collector_node = imitation_hub.data_collector_node:main',
            'inference_node = imitation_hub.inference_node:main',
            'train_bc = imitation_hub.train_bc:main'
        ],
    },
)
