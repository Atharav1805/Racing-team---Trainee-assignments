from setuptools import find_packages, setup

package_name = 'data_sync'

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
    maintainer='atharav',
    maintainer_email='atharav.sonawane.aryo@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "zed_auto = data_sync.zed_auto:main",
            "zed_manual = data_sync.zed_manual:main",
        ],
    },
)
