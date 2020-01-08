# gammanet_pytorch

How to start: 

The pipeline can be launched as symple as:

````
 python run.py config
````

The config folder contains the example for the meta-configuration that will be used during the experiment in .yaml files: 

 * datasets: 
    Configuration for the datasets (Test, train and validation). For example inside each .ymal file, the typical configuration metadata would be: 
    ````python
        name: BSDS500_crops
        import_prepath: data.BSDS500_100
        import_class: BSDS500
        images_path: /media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/images/test
        labels_path: /media/data_cifs/pytorch_projects/datasets/BSDS500_crops/data/groundTruth/test
        transform: 
            Resize:
            size: 320
        input: image
        target: label
    ````
 * exp: 
    Here the specifications for the experiment is given: Models to be used, logging directories, loss configurations, etc. 
* Model: 
    Here the configuration of each of the models that can be used is specified. Where to apply fgru units, attention, saliency, pretrained weights. etc. 

Please take a look at each of the files provided for reference. 

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