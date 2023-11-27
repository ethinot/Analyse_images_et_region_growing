CC = g++
PROJECT = bin/output
SRC = tp0.cpp
LIBS = `pkg-config --cflags --libs opencv4`
$(PROJECT) : $(SRC)
	$(CC) -Wall -g $(SRC) -o $(PROJECT) $(LIBS)
