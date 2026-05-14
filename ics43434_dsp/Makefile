CC      = gcc
CFLAGS  = -O2 -march=armv8-a -Wall -fPIC
LDFLAGS = -lm
LIB     = libaudio_dsp.so
SRC     = audio_dsp.c

all: $(LIB)

$(LIB): $(SRC) audio_dsp.h
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)
	@echo "--- DA DUC XONG DONG CO libaudio_dsp.so ---"

clean:
	rm -f $(LIB)
