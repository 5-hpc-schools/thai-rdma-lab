PROGRAMS := hello_mpi rank_control work_distribution pi_serial pi_mpi poly_serial poly_mpi \
            ping_pong collective_cost halo_exchange memory_stream
BUILD_DIR ?= build
CPPFLAGS ?= -Iinclude
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?=
LDLIBS ?= -lm

ifeq ($(origin CC),default)
CC := cc
endif

TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean print-env

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: src/%.c include/mpi_training.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

print-env:
	@echo "CC=$(CC)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "CFLAGS=$(CFLAGS)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@$(CC) --version 2>&1 | head -5 || true

clean:
	rm -rf build
