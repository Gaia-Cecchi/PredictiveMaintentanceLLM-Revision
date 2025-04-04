import subprocess
import os
import time
import signal

# Start server
server_process = subprocess.Popen(['python', 'server.py'])
time.sleep(10)

# Start clients
num_files = len(os.listdir('datasets'))
client_processes = []
for i in range(num_files):
    client_process = subprocess.Popen(['python', 'client.py', f'--client-id={i}'])
    client_processes.append(client_process)

# Define signal handler
def signal_handler(sig, frame):
    print("Terminating all processes...")
    # stop server and clients
    server_process.terminate()
    for client_process in client_processes:
        client_process.terminate()
    # wait for all processes to terminate
    server_process.wait()
    for client_process in client_processes:
        client_process.wait()
    exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Wait for all processes to terminate
server_process.wait()
for client_process in client_processes:
    client_process.wait()
