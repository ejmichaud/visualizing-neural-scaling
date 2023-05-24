
import random
import time

from itertools import product
import os
import sys

import numpy as np

model_names = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
]

if __name__ == '__main__':

    task_idx = int(sys.argv[1])

    model_name = model_names[task_idx]
    os.system(f"""python /om2/user/ericjm/visualizing-neural-scaling/scripts/eval_losses.py \
                                --save_dir /om/user/ericjm/results/visualizing-neural-scaling/eval_losses/ \
                                --model_name {model_name} \
                                --step 143000 \
                                --num_documents 50000 \
                                """)


