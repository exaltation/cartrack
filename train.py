import model
import common

weights_file = 'sample_weights.hd5'

model = model.get_training_model()
model.save_weights(weights_file)

detect_model = get_detect_model()
detect_model.load_weights(weights_file, by_name=True)

detect_model.summary()
