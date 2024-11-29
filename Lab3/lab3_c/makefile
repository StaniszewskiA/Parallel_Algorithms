CC = gcc

CFLAGS = -Wall -Wextra -O2 -g

SRCS = lab3_sequential.c

TARGET = lab3_sequential

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
