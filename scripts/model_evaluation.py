#9. Model Evaluation
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from model_training import load_data, train_model

# Load the test dataset
test_ds = load_data('data/dogs', subset='test')

# Load your trained model
model = tf.keras.models.load_model('models/imageclassifier.h5')

# Initialize metrics
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

# Evaluate the model on the test dataset
for batch in test_ds.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

# Print the evaluation results
print(f'Precision: {precision.result().numpy()}')
print(f'Recall: {recall.result().numpy()}')
print(f'Accuracy: {accuracy.result().numpy()}')
