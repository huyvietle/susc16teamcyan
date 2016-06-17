import SocketServer
from subprocess import call

class CommandsUDPHandler(SocketServer.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the UDP socket connected to the client
        data = self.request[0].strip()
        socket = self.request[1]
        print "{} wrote:".format(self.client_address[0])
        print data

        if data == "start":
            self.start_presentation()
        elif data == "stop":
            self.stop_presentation()
        elif data == "next":
            self.next_slide()
        elif data == "prev":
            self.prev_slide()

        # just send back the same data, but upper-cased
        socket.sendto(data.upper(), self.client_address)

    def start_presentation(self):
        call(["osascript","apple_scripts/start_presentation.scpt"])

    def stop_presentation(self):
        call(["osascript","apple_scripts/stop_presentation.scpt"])

    def next_slide(self):
        call(["osascript","apple_scripts/next_slide.scpt"])

    def prev_slide(self):
        call(["osascript","apple_scripts/prev_slide.scpt"])



if __name__ == "__main__":
    HOST, PORT = "localhost", 5555

    # Create the server, binding to localhost on port 9999
    server = SocketServer.UDPServer((HOST, PORT), CommandsUDPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()