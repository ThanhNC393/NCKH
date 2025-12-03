import os
import shutil
from setuptools import setup, find_packages

# Xóa các thư mục build, dist, và .egg-info nếu tồn tại
# for folder in ['build', 'dist', 'f1.egg-info']:
#     if os.path.exists(folder):
#         shutil.rmtree(folder)

setup(
    name='f1',
    version='2024.11.15',
    author='HaUI NCKH F1',
    description='Fuzzy clustering',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9'
)
