import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load dataset
def load_images(folder):
    images = []
    labels = []

    for label, emotion in enumerate(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']):
        emotion_folder = os.path.join(folder, emotion)
        for filename in os.listdir(emotion_folder):
            img = Image.open(os.path.join(emotion_folder, filename))
            img = img.resize((48, 48)).convert('L')
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=-1) # Add a new dimension for the color channel
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.eye(7)[labels]

X_train, y_train = load_images('./Train')
X_test, y_test = load_images('./Test')

# Create the CNN architecture
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 48, 48, 1], name='X')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7], name='y')

conv1 = tf.layers.conv2d(X, 32, 3, activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, 2, 2)

flat = tf.compat.v1.layers.flatten(pool3)
dense = tf.compat.v1.layers.Dense(128, activation=tf.nn.relu)(flat)

dropout = tf.compat.v1.layers.Dropout(0.5)(dense)
logits = tf.compat.v1.layers.Dense(7)(dropout)

# Define the loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

# Calculate accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the model
init = tf.compat.v1.global_variables_initializer()

n_epochs = 192
batch_size = 84

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch, y_batch = X_train[i:i + batch_size], y_train[i:i + batch_size]

            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
            # Calculate accuracy and loss for the test set
        test_acc, test_loss = sess.run([accuracy, loss], feed_dict={X: X_test, y: y_test})
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_acc}, Test Loss: {test_loss}")

    # Calculate accuracy and loss for the test set
    test_acc, test_loss = sess.run([accuracy, loss], feed_dict={X: X_test, y: y_test})
    print(f"Final : Epoch {epoch + 1}, Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss*100:.2f}%")

    # Save the model
    saver = tf.compat.v1.train.Saver()
    save_path = saver.save(sess, "./model/emotion_model.ckpt")
    print("Model saved in path: %s" % save_path)