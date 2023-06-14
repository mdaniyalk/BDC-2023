import numpy as np
import tensorflow as tf
import warnings

def character_accuracy_np(y_true, y_pred):
    """
    Calculate the character-based accuracy using NumPy.

    Parameters:
        y_true (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        float: Character-based accuracy.

    """
    accuracy = np.mean([np.mean(y_true[i] == y_pred[i]) for i in range(len(y_true))])
    return accuracy

def character_accuracy_tf(y_true, y_pred):
    """
    Calculate the character-based accuracy using TensorFlow.

    Parameters:
        y_true (tensorflow.Tensor): Ground truth labels.
        y_pred (tensorflow.Tensor): Predicted labels.

    Returns:
        tensorflow.Tensor: Character-based accuracy.

    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_flat, y_pred_flat), tf.float32))
    return accuracy

def character_accuracy(y_true, y_pred, lib='tf'):
    """
    Calculate the character-based accuracy.
    Tensorflow based function (lib='tf') should be used in model training.

    Parameters:
        y_true (numpy.ndarray or tensorflow.Tensor): Ground truth labels.
        y_pred (numpy.ndarray or tensorflow.Tensor): Predicted labels.
        lib (str, optional): Library type to use ('tf' for TensorFlow, 'np' for NumPy). Default is 'tf'.

    Returns:
        float or tensorflow.Tensor: Character-based accuracy.

    Raises:
        UserWarning: If an unknown library type is selected.

    """
    if lib == 'tf':
        return character_accuracy_tf(y_true, y_pred)
    elif lib == 'np':
        return character_accuracy_np(y_true, y_pred)
    else:
        warnings.warn('Warning: Unknown selected library type: Possible library type [\'tf\', \'np\']')
