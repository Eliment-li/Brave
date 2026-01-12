from setuptools import setup, find_packages

setup(
    name="brave_project",
    version="0.0.0",
    packages=find_packages(),  # 会自动发现 configs、train 等包
)