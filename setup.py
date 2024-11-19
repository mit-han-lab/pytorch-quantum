"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from setuptools import setup, find_packages

VERSION = {}  # type: ignore

with open("torchquantum/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)

if __name__ == "__main__":
    requirements = open("requirements.txt").readlines()
    requirements = [r.strip() for r in requirements]

    setup(
        name="torchquantum",
        version=VERSION["version"],
        description="Quantum Computing in PyTorch",
        long_description=open("README.md").read(),
        url="https://github.com/mit-han-lab/torchquantum",
        author="Shreya Chaudhary, Zhuoyang Ye, Jiannan Cao, Jessica Ding, Jiai Gu, Song Han, Zirui Li, Zhiding Liang, Pengyu Liu, Mohammadreza Tavasoli, Hanrui Wang",
        author_email="hanruiwang.hw@gmail.com",
        license="MIT",
        install_requires=requirements,
        extras_require={"doc": ["nbsphinx", "recommonmark"]},
        python_requires=">=3.5",
        include_package_data=True,
        packages=find_packages(),
    )
