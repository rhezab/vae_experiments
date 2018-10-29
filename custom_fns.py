from keras import backend as K

def none_loss(y_true, y_pred):
    return K.constant(0)
