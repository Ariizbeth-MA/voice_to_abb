from setuptools import find_packages, setup

package_name = 'robot_voice_commander'

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
    maintainer='arizbeth',
    maintainer_email='a01028529@tec.mx',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'voice_commander = robot_voice_commander.voice_commander_node:main',
            'moveit_executor = robot_voice_commander.moveit_executor_node:main',
            'benchmark = robot_voice_commander.benchmark:main',
            'benchmark_cpp = robot_voice_commander.benchmark_cpp:main',
            'enroll_speaker = robot_voice_commander.modules.speaker.enroll:main',
        ],
    },
)
