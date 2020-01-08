# gammanet_pytorch

# Organization of the Repo
|____run.py
|____layers
| |____hgru_base.py
| |______init__.py
| |____fgru_base.py
|____config
| |____dataset
| | |____BSDS500_test.yaml
| | |____BSDS500_val.yaml
| | |____BSDS500_100.yaml
| | |____BSDS500_train.yaml
| |____model
| | |____vgg_gammanet.yaml
| | |____sn_hgru.yaml
| | |____vgg_hgru.yaml
| |____exp
| | |____boundary_detection.yaml
|____experiments
| |______init__.py
| |____boundary_detection.py
| |____base.py
|____utils
| |____pt_utils.py
| |______init__.py
| |____py_utils.py
|____models
| |____vgg_16.py
| |____squeezenet.py
| |____vgg_gammanet.py
| |______init__.py
| |____sn_hgru.py
| |____vgg_hgru.py
|____README.md
|____ops
| |____metrics.py
| |____data_tools.py
| |____experiment_tools.pyc
| |______init__.py
| |____optimizers.py
| |______init__.pyc
| |____losses.py
| |____experiment_tools.py
| |____model_tools.py
|____data
| |______init__.py
| |____BSDS500_100.py