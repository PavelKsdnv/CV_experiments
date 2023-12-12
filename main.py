import cv2
import numpy as np

def display_webcam():
    # Open the webcam (use 0 for the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./mobilenet_iter_73000.caffemodel")

    # Set the confidence threshold
    confidence_threshold = 0.2

    while True:
        # Read a frame from the webcam
        ret, image = cap.read()

        # If reading the frame fails, break the loop
        if not ret:
            break

        # Resize the image to a fixed size
        resized_image = cv2.resize(image, (300, 300))

        # Convert the image to float32 and normalize
        normalized_image = (resized_image.astype(np.float32) / 127.5) - 1.0

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(normalized_image, 0.007843, (300, 300), 127.5)

        # Set the input to the model
        net.setInput(blob)

        # Run the forward pass to get predictions
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the confidence is above the threshold
            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                print("found something, class id: ", class_id)
                # Check if the detected object is a person (class ID 15)
                if class_id == 15:
                    # Get the coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the bounding box and label on the image
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Person Detection", image)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the webcam capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()
         


def display_video(file_path):
    # Open the video file
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Load the pre-trained model
    net = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./mobilenet_iter_73000.caffemodel")

    # Set the confidence threshold
    confidence_threshold = 0.2

    while True:
        # Read a frame from the video file
        ret, image = cap.read()

        # If the video is finished, break the loop
        if not ret:
            break

        # Resize the image to a fixed size
        resized_image = cv2.resize(image, (300, 300))

        # Convert the image to float32 and normalize
        normalized_image = (resized_image.astype(np.float32) / 127.5) - 1.0

        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(normalized_image, 0.007843, (300, 300), 127.5)

        # Set the input to the model
        net.setInput(blob)

        # Run the forward pass to get predictions
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the confidence is above the threshold
            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])

                # Check if the detected object is a person (class ID 15)
                if class_id == 15:
                    # Get the coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw the bounding box and label on the image
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(image, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Person Detection", image)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'your_video.mp4' with the path to your MP4 file
    video_file_path = 'testVideo.mp4'
    
    # Call the function to display the video
    # display_video(video_file_path)

    display_webcam()
