from setuptools import setup, find_packages

setup(
    name="aipiler",
    version="0.1.0",
    url="https://github.com/Aipiler/Aipiler",  # 项目地址
    package_dir={"": "python"},  # 指定根目录下的python文件夹
    packages=find_packages(where="python"),  # 自动发现python目录下的包
    python_requires=">=3.12.8",  # 指定Python版本要求
    install_requires=[
        "torch==2.7.0",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            # 如果需要生成命令行工具，例如：
            # "aipiler-cli = Aipiler.cli:main",
        ],
    },
)