from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import subprocess

def install_dacbench():
    # Path to the contained library
    contained_library_path = os.path.join(os.path.dirname(__file__), 'DACBench/')
    print(f'Installing DACBench from {contained_library_path}')
    # Install the contained library
    subprocess.check_call(['pip', 'install', '-e', contained_library_path])
    print('Installed DACBench. Continuing with the installation of the rest of the package.')

class FirstInstallDACBench(install):
    """Customized setuptools install command - install DACBench first which should be at the root of the candidDAC directory."""
    def run(self):
        install_dacbench()
        # Proceed with the normal installation
        install.run(self)

class FirstInstallDACBenchDevelop(develop):
    """Customized setuptools development install command - install DACBench first which should be at the root of the candidDAC directory."""
    def run(self):
        install_dacbench()
        develop.run(self)


setup(
    name='candid_dac',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gymnasium==0.29.1',
        'hydra-core==1.3.2',
        'numpy==1.24.2',
        'stable-baselines3==2.1.0',
        'torch==2.1.0',
        'tqdm==4.66.1',
        'wandb==0.15.12',
    ],
    cmdclass={'install': FirstInstallDACBench, 'develop': FirstInstallDACBenchDevelop},
    entry_points={
        'console_scripts': [
            # 'script_name=module:function'
        ],
    },
    author='Philipp Bordne',
    author_email='bordnep@cs.uni-freiburg.com',
    description='Algorithms and policies for the CANDID DAC paper.',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    url='https://github.com/PhilippBordne/candidDAC',
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    python_requires='>=3.10, <3.11',
)
