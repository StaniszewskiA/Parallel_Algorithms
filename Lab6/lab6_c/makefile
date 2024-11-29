CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99 -lm
LDFLAGS = -mconsole  

TARGET = lab6

SRC = lab6.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

rebuild: clean all

.PHONY: all clean rebuild
