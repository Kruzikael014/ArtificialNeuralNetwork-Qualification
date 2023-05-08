import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
graph = tf.Graph()
with graph.as_default():
    sess = tf.compat.v1.Session()
    with sess.as_default():
        saver = tf.compat.v1.train.import_meta_graph('./model/emotion_model.ckpt.meta')
        saver.restore(sess, './model/emotion_model.ckpt')

        face_cascade = cv2.CascadeClassifier('./scripts/haarcascade_frontalface_default.xml')

        # Get the input and output tensors
        X = graph.get_tensor_by_name('X:0')
        logits = graph.get_tensor_by_name('dense_1/BiasAdd:0')

        # Set up the video capture
        cap = cv2.VideoCapture(0)

        # Define the emotion labels
        labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:

                face = gray[y:y+h, x:x+w]

                # Resize the frame to 48x48
                resized = cv2.resize(face, (48, 48))

                # Reshape the frame to match the input shape of the model
                input_data = np.reshape(resized, (1, 48, 48, 1))

                # Normalize the input data
                input_data = input_data / 255.0

                # Make the prediction
                prediction = sess.run(tf.argmax(logits, 1), feed_dict={X: input_data})

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                print(f'{prediction[0]} - {labels[prediction[0]]}')

                # Display the predicted label on the screen
                cv2.putText(frame, labels[prediction[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()