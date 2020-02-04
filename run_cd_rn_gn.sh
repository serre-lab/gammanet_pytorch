CUDA_VISIBLE_DEVICES=0,2,4,5,6 python run.py config/exp/contour_detections_resnet_train \
                                            dataloader.kwargs.batch_size=4 \
                                            model.args.base_ff.args.freeze_layers=True