## This is the code of AI Master Preoject in University of Groningen
This project is developed based on [AlphaZero General(https://github.com/suragnair/alpha-zero-general)
## How to run the draughts project

### Train the model
Baseline method:

`python -u main.py -m residual_baseline -c OUTPUT_DIR-`

Three-stage method:

`python -u main.py -m residual_three -t -c OUTPUT_DIR-`

MPV method:

`python -u main.py -m residual_mpv -b 8 -n 10 -c OUTPUT_DIR-`

Three-stage and MPV hybrid method:

`python -u main.py -m residual_mvp_three -b 8 -n 10 -t -c OUTPUT_DIR-`

Three-stage method with larger CNN:

`python -u main.py -m residual_three -t -g -c OUTPUT_DIR-`

Three-stage and MPV hybrid method with larger CNN:

`python -u main.py -m residual_mvp_three -b 8 -n 10 -t -g -c OUTPUT_DIR-`

### Play against the model

change the checkpoint file path in `pit.py ` and run `python pit.py`

