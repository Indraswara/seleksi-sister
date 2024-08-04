# Variables
CC = gcc
CFLAGS = -Wall -Wextra -Wno-unused
OBJDIR = bin
TARGET = server
SRC = src
# Source files
SRCS = $(wildcard $(SRC)/*.c)
OBJS = $(patsubst $(SRC)/%.c, $(OBJDIR)/%.o, $(SRCS))

# Default target
all: $(OBJDIR) $(TARGET)

# Create bin directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
$(OBJDIR)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean