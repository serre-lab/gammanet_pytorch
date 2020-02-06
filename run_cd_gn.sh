CUDA_VISIBLE_DEVICES=3,4,5,6 python run.py config/exp/contour_detection \
                                            dataloader.kwargs.batch_size=4 \
                                            model.args.base_ff.args.freeze_layers=false