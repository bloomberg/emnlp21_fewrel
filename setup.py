from setuptools import setup, find_packages

setup(name='FewRel',
      version='2.0',
      description="Modifications on 'A Large-Scale Few-Shot Relation Extraction Dataset' - https://github.com/thunlp/FewRel",
      url='https://bbgithub.dev.bloomberg.com/sbrody18/FewRel',
      author='Sam Brody',
      author_email='sbrody18@bloomberg.net',
      license='',
      packages=find_packages(),
      include_package_data=True,
      package_data={'': ['data/*.json']},    
      install_requires=['scikit-learn',
                        'torch==1.6.0',
                        'transformers==3.4.0',
                        ],
      zip_safe=False)
