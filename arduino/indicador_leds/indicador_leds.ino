/*
  Indicador de LEDs para el reconocimiento facial (Arduino Uno).

  Conexión (cada LED con resistencia de 220 ohm en serie):
    D8  -> LED persona_1 -> GND
    D9  -> LED persona_2 -> GND
    D10 -> LED persona_3 -> GND

  Protocolo serie a 9600 baudios, un carácter por orden:
    '0' apaga todos | '1' enciende persona_1 | '2' persona_2 | '3' persona_3

  Seguridad: si pasan más de 3 s sin recibir ninguna orden (por ejemplo, la aplicación
  se cerró o el cable se soltó), se apagan todos los LED.
*/

const int PIN_LED[3] = {8, 9, 10};
const unsigned long TIEMPO_SIN_ORDEN_MS = 3000;

unsigned long ultimaOrden = 0;

void apagarTodos() {
  for (int i = 0; i < 3; i++) digitalWrite(PIN_LED[i], LOW);
}

void encenderSolo(int indice) {
  for (int i = 0; i < 3; i++) digitalWrite(PIN_LED[i], i == indice ? HIGH : LOW);
}

void setup() {
  for (int i = 0; i < 3; i++) pinMode(PIN_LED[i], OUTPUT);
  apagarTodos();
  Serial.begin(9600);
  ultimaOrden = millis();

  // Prueba de arranque: cada LED parpadea una vez
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED[i], HIGH);
    delay(150);
    digitalWrite(PIN_LED[i], LOW);
  }
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '0') {
      apagarTodos();
      ultimaOrden = millis();
    } else if (c >= '1' && c <= '3') {
      encenderSolo(c - '1');
      ultimaOrden = millis();
    }
  }

  if (millis() - ultimaOrden > TIEMPO_SIN_ORDEN_MS) {
    apagarTodos();
  }
}
