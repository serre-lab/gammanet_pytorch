CUDA_VISIBLE_DEVICES=0,7 python run.py config/exp/contour_detection \
                                            dataloader.kwargs.batch_size=2 \
                                            model.args.base_ff.args.freeze_layers=false