EXP_NAME="vgg_gn_cd_160"

nohup \
python run.py   config/exp/contour_detection \
                name=$EXP_NAME \
                parallel=true \
                gpus=4 \
                val_freq=1 \
                save_freq=1 \
                dataloader.kwargs.batch_size=12 \
                model.args.base_ff.args.freeze_layers=false \
                trainset=-dataset.BSDS500_train_aug \
                valset=-dataset.BSDS500_val \
                > $EXP_NAME.log 2>&1 &

                # track_grads.grad_flow.freq=1 \
                # trainset.transform.Resize.size=250 \
                # valset.transform.Resize.size=250 \
                
                