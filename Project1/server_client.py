import socket
import pickle
import threading
import concurrent.futures

class ServerClient:
    def __init__(self, SERVER_IP="172.20.10.2", SERVER_PORT=2238, CLIENT_IP="172.20.10.4", CLIENT_PORT=2238, SAVE_AUDIO_LOCATION='Recieved Data', SAVE_IMAGE_LOCATION = 'Recieved Data'):
        self.SERVER_IP = SERVER_IP
        self.SERVER_PORT = SERVER_PORT
        self.CLIENT_IP = CLIENT_IP
        self.CLIENT_PORT = CLIENT_PORT
        self.SAVE_AUDIO_LOCATION = SAVE_AUDIO_LOCATION
        self.SAVE_IMAGE_LOCATION = SAVE_IMAGE_LOCATION
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    def run_server(self):
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the server address and port
        server_socket.bind((self.SERVER_IP, self.SERVER_PORT))

        # Listen for incoming connections
        server_socket.listen(1)

        print("Server is listening on", self.SERVER_IP, "port", self.SERVER_PORT)

        while True:
            # Accept a connection from a client
            client_socket, client_address = server_socket.accept()
            print("Connection established with", client_address)

            # Receive the combined audio and image data
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk

            # Unpickle the received data
            audio_data, image_data = pickle.loads(data)

            print("Audio and image data received successfully.")

            # Save the received audio and image to the specified locations
            with open(self.SAVE_AUDIO_LOCATION + "/received_audio.wav", "wb") as audio_file:
                audio_file.write(audio_data)

            with open(self.SAVE_IMAGE_LOCATION + "/received_image.jpg", "wb") as image_file:
                image_file.write(image_data)

            # Close the client socket
            client_socket.close()

    def send_data(self, AUDIO_FILE_PATH = 'Send Data\output.wav', IMAGE_FILE_PATH = 'Send Data\captured_image.jpg'):
        # Read the audio file
        with open(AUDIO_FILE_PATH, 'rb') as audio_file:
            audio_data = audio_file.read()

        # Read the image file
        with open(IMAGE_FILE_PATH, 'rb') as image_file:
            image_data = image_file.read()

        # Combine audio and image data into a tuple
        data = (audio_data, image_data)

        # Pickle the combined data
        data_pickle = pickle.dumps(data)

        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect to the server
        client_socket.connect((self.CLIENT_IP, self.CLIENT_PORT))

        # Send the pickled data to the server
        client_socket.sendall(data_pickle)

        # Close the socket
        client_socket.close()

    def start_client_thread(self, AUDIO_FILE_PATH = 'Send Data\output.wav', IMAGE_FILE_PATH = 'Send Data\captured_image.jpg'):
        # Start a new thread to send data
        self.executor.submit(self.send_data, AUDIO_FILE_PATH, IMAGE_FILE_PATH)

# Create a ServerClient object
sc = ServerClient()

# Start the server in a separate thread
server_thread = threading.Thread(target=sc.run_server)
server_thread.start()

# In the main thread, wait for the user to enter a command
while True:
    command = input("")
    if command == "send":
        # When you want to send data from the client, call the start_client_thread method
        sc.start_client_thread()
    elif command == "quit":
        break
