import socket
import cv2
import pickle
import struct
import threading
import time

# Server Configuration
HOST = '0.0.0.0'  # Bind To All Interfaces
PORT = 9999

def send_periodic_message(client_socket):
    """Thread to send 'Server Connection Established' every 10 seconds."""
    while True:
        try:
            message = "Server Connection Established"
            client_socket.send(message.encode('utf-8'))
            print(f"Sent: {message}")
            time.sleep(10)
        except Exception as e:
            print(f"Error sending message: {e}")
            break

def handle_client(client_socket, addr):
    """Handle a single client connection."""
    print(f"Connection from {addr}")
    send_thread = threading.Thread(target=send_periodic_message, args=(client_socket,))
    send_thread.daemon = True
    send_thread.start()

    data = b""
    payload_size = struct.calcsize("Q")

    try:
        while True:
            # Receive Frame Size
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    raise ConnectionError("Client disconnected.")
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # Receive Frame Data
            while len(data) < msg_size:
                data += client_socket.recv(4096)

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Deserialize And Decode Frame
            img_encoded = pickle.loads(frame_data)
            frame = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

            # Display Frame
            cv2.imshow('Server - Received Webcam Feed', frame)

            # Quit 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during reception from {addr}: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()
        print(f"Connection closed for {addr}")

def main():
    # Create Server Socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.settimeout(30)  # Timeout For Accepting Connections
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Server listening on {HOST}:{PORT}...")

    try:
        while True:
            try:
                client_socket, addr = server_socket.accept()
                handle_client(client_socket, addr)
                print("Waiting for new connection...")
            except socket.timeout:
                print("No new connections; still listening...")
                continue
            except Exception as e:
                print(f"Error accepting connection: {e}")
                continue
    except KeyboardInterrupt:
        print("Server stopped by user.")
    finally:
        server_socket.close()
        cv2.destroyAllWindows()
        print("Server closed.")

if __name__ == "__main__":
    main()