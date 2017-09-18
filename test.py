import model as models

def print_to(model, txtfile):
    with open(txtfile,'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

print_to(models.get_training_model(), 'training_model.txt')
print_to(models.get_detect_model(123), 'detect_model.txt')
