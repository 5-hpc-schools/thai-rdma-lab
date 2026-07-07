PROGRAMS := hello_environment compute_pi memory_stream ping_pong allreduce_benchmark halo_exchange
BUILD_DIR ?= build
CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?=
LDLIBS ?=

ifeq ($(origin CXX),default)
CXX := CC
endif

TARGETS := $(addprefix $(BUILD_DIR)/,$(PROGRAMS))

.PHONY: all clean print-env

all: $(TARGETS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%: src/%.cpp include/mpi_training.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

print-env:
	@echo "CXX=$(CXX)"
	@echo "CPPFLAGS=$(CPPFLAGS)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@$(CXX) --version 2>&1 | head -5 || true

clean:
	rm -rf build
