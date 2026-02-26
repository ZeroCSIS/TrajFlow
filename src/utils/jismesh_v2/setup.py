#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import re
from setuptools import setup
from setuptools import find_packages

with io.open('README.md', 'rt', encoding='utf8') as f:
    readme = f.read()

with io.open('requirements.txt', 'rt', encoding='utf8') as f:
    requirements = f.read().split('\n')

with io.open('test_requirements.txt', 'rt', encoding='utf8') as f:
    test_requirements = f.read().split('\n')

with io.open('jismesh/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setup(name='jismesh',
      version=version,
      packages=find_packages(),
      description='Utilities for the Japanese regional grid system defined in Japanese Industrial Standards (JIS X 0410 地域メッシュ).',
      long_description_content_type='text/markdown',
      long_description=readme,
      keywords = ['mesh', 'grid', 'meshcode', 'mesh code', 'JIS X 0410'],
      author='Haruki Nishikawa',
      author_email='harukinishikawa84@hotmail.com',
      url='https://github.com/hni14/jismesh',
      download_url='https://github.com/hni14/jismesh/archive/v{}.tar.gz'.format(version),
      license = 'MIT',
      platforms='any',
      install_requires=requirements,
      extras_require={
         'test': test_requirements,
         ':python_version < "3.0"': [
            'functools32',
         ],
      },
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        ],
     )
