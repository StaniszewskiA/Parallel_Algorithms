CC = gcc

CFLAGS = -Wall -Wextra -O2 -g

SRCS = lab1_sequential.c

TARGET = lab1_sequential

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
