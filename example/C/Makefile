.PHONY: all run clean

all: test_c

test_c: main.c
	clang main.c -I../../Sources/libvevc/include -L../../.build/release -llibvevc -o test_c

run: test_c
	DYLD_LIBRARY_PATH=../../.build/release ./test_c

clean:
	rm -f test_c
