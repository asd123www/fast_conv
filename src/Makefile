
CC = gcc
CFLAGS = -I$(INCDIR) -mavx -mavx2 -mfma -msse -msse2 -msse3 -Wall -O3 -std=c99

INCDIR = ../include
BINDIR = ../bin
OBJDIR = ../obj

ALL = basic_conv2d

$(OBJDIR)/%.o: %.c
	@[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CC) -c -o $@ $< $(CFLAGS)

all: $(ALL)

basic_conv2d: $(patsubst %, $(OBJDIR)/%, basic_conv2d.o)
	@[ -d $(BINDIR) ] || mkdir $(BINDIR)
	$(CC) -o $(BINDIR)/$@ $^ $(CFLAGS)

run-%: $(BINDIR)/%
	$^

runall: all
	@echo -e
	@for exe in $(ALL); do \
		echo "$(BINDIR)/$$exe"; \
		$(BINDIR)/$$exe; \
		echo -e; \
	done


.PHONY: clean
clean:
	-@rm -rf $(BINDIR) $(OBJDIR)
	@echo "- Cleaned Directory '/'"