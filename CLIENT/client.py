import socket
import cv2
import pickle
import struct
import threading
import time

# Server Configuration
HOST = 'SERVER_IP'  # Replace With Server's Public IP
PORT = 9999

def receive_messages(client_socket):
    """Thread to receive messages from server."""
    client_socket.settimeout(10)  # Set Timeout For Receiving Messages
    while True:
        try:
            message = client_socket.recv(1024).decode('utf-8', errors='ignore')
            if message:
                print(f"Received from server: {message}")
            else:
                print("Server disconnected.")
                break
        except socket.timeout:
            print("Receive timeout; server may be slow or disconnected.")
            continue
        except Exception as e:
            print(f"Error receiving message: {e}")
            break

def main():
    # Create Client Socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)  # Timeout For Initial Connection
    try:
        client_socket.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return

    client_socket.setblocking(False)

    # Start Thread For Receiving Messages
    recv_thread = threading.Thread(target=receive_messages, args=(client_socket,))
    recv_thread.daemon = True
    recv_thread.start()

    # Capture Webcam Feed
    cap = cv2.VideoCapture(0)  # 0 = Default Camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        client_socket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Resize Frame
            frame = cv2.resize(frame, (640, 380))

            # Display Frame Locally
            cv2.imshow('Client - Webcam Feed', frame)

            # Quit on 'q' Key Press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Serialize Frame: Compress to JPEG and Pickle
            _, img_encoded = cv2.imencode('.jpg', frame)
            data = pickle.dumps(img_encoded)

            # Pack Message Size and Send
            message_size = struct.pack("Q", len(data))
            try:
                client_socket.sendall(message_size + data)
            except Exception as e:
                print(f"Error sending frame: {e}")
                break

            # Control FPS
            time.sleep(0.03)  # ~30 FPS

    except KeyboardInterrupt:
        print("Client stopped by user.")
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        cap.release()
        client_socket.close()
        cv2.destroyAllWindows() 
        print("Connection closed.")

if __name__ == "__main__":
    main()