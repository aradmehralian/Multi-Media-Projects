#include <UIPEthernet.h>
#include <UIPEthernetServer.h>
#include <UIPEthernetClient.h>
#include <WebSocketsServer.h>

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED }; // Replace with your own MAC address
IPAddress ip(192, 168, 1, 177); 
EthernetServer server(80);
WebSocketsServer webSocket = WebSocketsServer(81);

void setup() {
  Ethernet.begin(mac, ip);
  server.begin();
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
}

void loop() {
  webSocket.loop();
  EthernetClient client = server.available();
  // Read LUX sensor data
  int luxValue = analogRead(A0);
  // Send LUX data over WebSocket
  webSocket.broadcastTXT(String(luxValue));
  delay(1000); // Adjust delay as needed
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
}
