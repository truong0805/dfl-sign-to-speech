def get_weights(model):
    return [w.tolist() for w in model.get_weights()]

def load_weights(model, weights):
    import numpy as np
    model.set_weights([np.array(w) for w in weights])