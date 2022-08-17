import numpy as np

def fdr(p_vals) -> np.ndarray:
    from scipy.stats import rankdata
    
    if type(p_vals) == list:
        p_vals = np.array(p_vals)

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr