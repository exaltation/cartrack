import model as models
import common
from keras.models import load_model

trained_model_file = 'model.h5'

models.get_training_model.save(trained_model_file)

trained_model = load_model(trained_model_file)

detect_model = models.get_detect_model(trained_model.get_weights())
detect_model.summary()
