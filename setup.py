from setuptools import setup, find_packages

setup(name='FewRel',
      version='2.0',
      description="Modifications on 'A Large-Scale Few-Shot Relation Extraction Dataset' - https://github.com/thunlp/FewRel",
      url='https://github.com/bloomberg/emnlp2021_fewrel',
      author='Sam Brody',
      author_email='sbrody18@bloomberg.net',
      packages=find_packages(),
      include_package_data=True,
      package_data={'': ['data/*.json']},    
      install_requires=['scikit-learn',
                        'torch==1.6.0',
                        'transformers==4.30.0',
                        'ujson',
                        'wikipedia2vec'
                        ],
      zip_safe=False)
