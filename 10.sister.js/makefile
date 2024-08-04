# Variables
CC = gcc
CFLAGS = -Wall -Wextra
OBJDIR = bin
TARGET = server

# Source files
SRCS = server.c controller.c parser.c route.c
OBJS = $(patsubst %.c, $(OBJDIR)/%.o, $(SRCS))

# Default target
all: $(OBJDIR) $(TARGET)

# Create bin directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source files to object files
$(OBJDIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean