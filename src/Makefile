TARGET?=tinn
SRC=$(wildcard *.c)
OBJ=$(SRC:.c=.o)
CC = gcc
CFLAGS?= -Xprologue_epilogue_hooks_noinline -DGHS_FEE_COLLECT -G
LDLIBS=-lm -lghseagle
DESTDIR?=/usr/local/bin

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDLIBS)
	@-rm -f *.o 

clean:
	@-rm -f $(TARGET) $(OBJ) *.dbo *.dla *.dnm *.dle

train.o: Tinn.h
Tinn.o: Tinn.h
