from dso import DeepSymbolicOptimizer_PDE

#outer packages
import collections
import time
import os
import sys
import pickle
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__  == "__main__":
    warnings.filterwarnings('ignore', 'Intel MKL ERROR')
    pde = sys.argv[1] if len(sys.argv) > 1 else "KdV"
    folder = sys.argv[2] if len(sys.argv) > 2 else "MODE1"

    # build model by passing the path of user-defined config file. 
    model = DeepSymbolicOptimizer_PDE(f"./dso/config/{folder}/config_pde_{pde}.json")
    
    # model training
    start = time.time()
    result = model.train()
    cost_time=time.time() - start

    print("cost time : ",cost_time)

    # save_path = model.config_experiment["save_path"]
    # summary_path = os.path.join(save_path, "summary.csv")

    # with open(f'{pde}.pkl', 'wb') as f:
    #     pickle.dump(result, f)
    best_program = result["program"]
    y_true = best_program.task.ut
    y_pred = best_program.execute(best_program.task.u, best_program.task.x, best_program.task.ut)[0]
    r2 = 1 - ((y_pred - y_true)**2).sum() / max(((y_true - y_true.mean())**2).sum(), 1e-12)
    print(f"R2: {float(r2):.4f}")
        
