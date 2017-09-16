import model as models
import common
from keras.models import load_model

def trainprint(s):
    with open('train.txt','w+') as f:
        print(s, file=f)

def detectprint(s):
    with open('detect.txt','w+') as f:
        print(s, file=f)


trained_model_file = 'model.h5'

models.get_training_model().save(trained_model_file)

trained_model = load_model(trained_model_file)
trained_model.summary(print_fn=trainprint)

detect_model = models.get_detect_model(trained_model.get_weights())
detect_model.summary(print_fn=detectprint)

# trained_model = models.get_training_model()
# detect_model = models.get_detect_model(123)
#
# fc_1 = trained_model.get_layer('fc_1').get_weights()
# conv_fc_1 = detect_model.get_layer('conv_fc_1').get_weights()
#
# print(fc_1[0].shape)
# print(conv_fc_1[0].shape)
