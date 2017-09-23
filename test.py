import model as models

weights_file = 'model_weights_fc1_7.h5'

def print_to(model, txtfile):
    with open(txtfile,'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

# print_to(models.get_training_model(), 'training_model.txt')
print_to(models.get_detect_model(weights_file), 'detect_model.txt')
