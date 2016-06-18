import socket
import sys
import time

HOST, PORT = "10.100.10.194", 5555
data = " ".join(sys.argv[1:])

# Create a socket (SOCK_STREAM means a UDP socket)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cmd_list = ["start", "next", "play test"]

try:
	while len(cmd_list) != 0:
		cmd = cmd_list.pop(0)
		sock.sendto(cmd + "\n", (HOST, PORT))
	    # Receive data from the server and shut down
		received = sock.recv(1024)
		print "Sent:     {}".format(cmd)
		print "Received: {}".format(received)
		time.sleep(1.5)

finally:	
	sock.close()