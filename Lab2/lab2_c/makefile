CC = gcc
CFLAGS = -Wall -g -lm

SRC = lab2.c
OBJ = lab2.o

EXE = lab2_program

all: $(EXE)

$(EXE): $(OBJ)
	$(CC) $(OBJ) -o $(EXE)

$(OBJ): lab2.c
	$(CC) $(CFLAGS) -c lab2.c

clean:
	rm -f $(OBJ) $(EXE)

run: $(EXE)
	./$(EXE)

.PHONY: all clean run
