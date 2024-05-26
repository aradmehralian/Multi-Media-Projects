import cv2

def capture_image(output_file="Send Data/captured_image.jpg"):
    # Create a VideoCapture object to access the camera
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    # Create a window to display the live footage
    cv2.namedWindow("Live Footage", cv2.WINDOW_NORMAL)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Display the live footage in the created window
            cv2.imshow("Live Footage", frame)

            # If 'c' is pressed on the keyboard, capture the image and break the loop
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Save the captured frame as an image
                cv2.imwrite(output_file, frame)
                print("Image captured and saved as", output_file)
                break
        else:
            print("Error: Failed to capture image.")
            break

    # Release the VideoCapture object
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()

capture_image()
