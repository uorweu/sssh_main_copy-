import logging
try:
    from gpiozero import LED
except ImportError:
    LED = None

log = logging.getLogger("leds")

class LedController:
    """
    Manages 3 LEDs (Green, Yellow, Red) to visualize the current noise level.
    LEDs respond directly to real-time dB levels — no FSM/cooldown involved.

    Thresholds:
      - Green:  dB < yellow_threshold  (quiet)
      - Yellow: yellow_threshold <= dB < red_threshold  (moderate)
      - Red:    dB >= red_threshold  (loud)
    """
    def __init__(self, pin_green: int = 17, pin_yellow: int = 27, pin_red: int = 22,
                 yellow_threshold: float = 55.0, red_threshold: float = 75.0):
        self.enabled = False
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold

        if LED is None:
            log.warning("gpiozero not installed or not on Raspberry Pi. LEDs disabled.")
            return

        try:
            self.led_green = LED(pin_green)
            self.led_yellow = LED(pin_yellow)
            self.led_red = LED(pin_red)
            self.enabled = True
            log.info("LEDs initialized on GPIO pins %d (G), %d (Y), %d (R)",
                     pin_green, pin_yellow, pin_red)
            # Start with green (quiet)
            self.led_green.on()
        except Exception as e:
            log.error("Failed to initialize LEDs: %s", e)
            self.enabled = False

    def update(self, spl_db: float):
        """
        Update LEDs based on the current real-time dB level.
        Reacts instantly — no cooldown, no FSM.
        """
        if not self.enabled:
            return

        # Turn all off first
        self.led_green.off()
        self.led_yellow.off()
        self.led_red.off()

        # Light up based on dB level
        if spl_db >= self.red_threshold:
            self.led_red.on()
        elif spl_db >= self.yellow_threshold:
            self.led_yellow.on()
        else:
            self.led_green.on()

    def close(self):
        """Cleanup / turn off LEDs when shutting down."""
        if self.enabled:
            self.led_green.off()
            self.led_yellow.off()
            self.led_red.off()
            self.led_green.close()
            self.led_yellow.close()
            self.led_red.close()
            self.enabled = False

