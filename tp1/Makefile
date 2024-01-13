CC = g++
PROJECT = bin/main.out
SRC = main.cpp
LIBS = `pkg-config --cflags --libs opencv4`
$(PROJECT) : $(SRC)
	$(CC) -Wall -g -std=c++17 -pthread $(SRC) -o $(PROJECT) $(LIBS)