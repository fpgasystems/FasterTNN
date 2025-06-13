.PHONY: autotune timeall time plot build clean

# Source files
TNNS_SRC:=$(wildcard src/tnns/*.c | wildcard src/tnns/gemm-variants/*.c | src/*.c)
TNNS_HDR:=$(wildcard include/tnns/*.h | include/timing/*.h | include/*.h)
TNNS_OUT:=$(patsubst src/%.c,out/%.o,$(TNNS_SRC))

# Timing source files
TIME_LAYERS_SRC:=src/timing_layers.c src/utils.c out/tnns/data.o
TIME_GEMM_SRC:=src/timing_gemm.c src/utils.c out/tnns/data.o

# OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Compiler flags
ifeq ($(UNAME_S),Linux) # For Linux
	ifeq ($(UNAME_M),x86_64) # For x86_64 chips
		CCFLAGS:= -O3 -DAVX2 -Iinclude -Wall -Wextra -pedantic -march=core-avx2 -mavx2 -mpopcnt -g
		CPPFLAGS:= -msse4.1 -msse4.2 -mavx2 -mpopcnt -march=core-avx2 -O3 -Iinclude
		TASKSET_CMD:=taskset -c 0
		ISA := _avx2
	else 
	ifeq ($(UNAME_M),aarch64) # For ARM chips
		CCFLAGS:= -O3 -DARM_NEON -Iinclude -Wall -Wextra -pedantic -Wno-unused-function -flax-vector-conversions
		CPPFLAGS:= -std=c++2b -march=native -O3 -Iinclude
		TASKSET_CMD:=taskset -c 0
		ISA := _arm_neon
	endif
	endif
endif
ifeq ($(UNAME_S),Darwin) # For macOS	
	BRAND := $(shell sysctl -n machdep.cpu.brand_string)
	TASKSET_CMD:=
	ifeq ($(BRAND),Apple M1 Pro) # For Apple M1 Pro
		CCFLAGS:= -O3 -DARM_NEON -Iinclude -Wall -Wextra -pedantic -mcpu=apple-m1 -Wno-unused-function
		CPPFLAGS:= -std=c++2b -march=native -O3 -Iinclude
		ISA := _arm_neon
	else 
	ifeq ($(BRAND),Apple M2) # For Apple M2
		CCFLAGS:= -O3 -DARM_NEON -Iinclude -Wall -Wextra -pedantic -mcpu=apple-m1 -Wno-unused-function
		CPPFLAGS:= -std=c++2b -O3 -Iinclude
		ISA := _arm_neon
	endif
	endif
endif


# Timing targets
VERSIONS := baseline final
TIME_MODELS := tnn tbn btn

# Single-layer timing
time_layers_all: $(foreach l,$(TIME_MODELS),time_layers_$(l))
time_layers_%: build out/measurements
	@for v in $(VERSIONS); do \
		$(TASKSET_CMD) out/$$v $$v $* layer; \
	done

# End-to-end (multi-layer) timing
time_e2e_all: $(foreach l,$(TIME_MODELS),time_e2e_$(l))
time_e2e_%: build out/measurements
	@for v in $(VERSIONS); do \
		$(TASKSET_CMD) out/$$v $$v $* e2e; \
	done

# Test targets
TEST_MODELS := tnn tbn btn bnn

test_all: $(foreach l,$(TEST_MODELS),test_$(l))
test_%: build out/measurements
	@for v in $(VERSIONS); do \
		$(TASKSET_CMD) out/$$v $$v $* test; \
	done


# Build targets
build: out/tnn out/baseline out/final

out/baseline: $(TIME_LAYERS_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_naive.o 
	gcc $(CCFLAGS) $(TIME_LAYERS_SRC) out/tnns/tnns_cpu_naive.o -o out/baseline

out/final: $(TIME_LAYERS_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p2.o out/tnns/mm_p2_final_tnn$(ISA).o out/tnns/mm_p2_final_tbn$(ISA).o out/tnns/mm_p2_final_btn$(ISA).o out/tnns/mm_p2_final_bnn$(ISA).o out/tnns/binarize2row$(ISA).o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_LAYERS_SRC) out/tnns/tnns_cpu_mm_p2.o out/tnns/mm_p2_final_tnn$(ISA).o out/tnns/mm_p2_final_tbn$(ISA).o out/tnns/mm_p2_final_btn$(ISA).o out/tnns/mm_p2_final_bnn$(ISA).o out/tnns/binarize2row$(ISA).o out/tnns/ternarize2row$(ISA).o -o out/final


# Create output directories 
out/tnn:
	mkdir -p out/tnn
	mkdir -p out/tnns/gemm-variants

# Compile general source files into object files
out/tnns/%.o: src/tnns/%.c $(TNNS_HDR) | out/tnns/
	gcc $(CCFLAGS) -c src/tnns/$*.c -o $@

# Compile GEMM variant source files into object files
out/tnns/gemm-variants/%.o: src/tnns/gemm-variants/%.c $(TNNS_HDR) | out/tnns/gemm-variants/
	gcc $(CCFLAGS) -c src/tnns/gemm-variants/$*.c -o $@

# Create measurement output directories 
out/measurements:
	mkdir -p $@
	chmod 774 out
	chmod -R 774 $@

out/measurements_gemm:
	mkdir -p out/measurements/gemm
	chmod 774 out
	chmod 774 out/measurements
	chmod -R 774 out/measurements/gemm

# Save current measurements for archiving
save_measurements:
	mkdir -p out-saved/measurements/gemm
	cp -r out/measurements/* out-saved/measurements/
	cp -r out/measurements/gemm/* out-saved/measurements/gemm/

# Clean all build output
clean:
	rm -rf out



# GEMM variants: only for AVX2
build_gemm: out/naive_gemm out/p0_blocked_gemm out/p0_inst_gemm out/p0_lib_gemm out/p1_2pc_gemm out/p1_1pc_gemm out/p1_csas_gemm out/p2_gemm

# Time all GEMM variants
time_gemm: build_gemm out/measurements_gemm
	$(TASKSET_CMD) out/naive_gemm baseline
	$(TASKSET_CMD) out/p0_blocked_gemm blocked
	$(TASKSET_CMD) out/p0_inst_gemm popcnt64
	$(TASKSET_CMD) out/p1_2pc_gemm popcnt256
	$(TASKSET_CMD) out/p1_1pc_gemm harley-seal
	$(TASKSET_CMD) out/p1_csas_gemm opt-csas
	$(TASKSET_CMD) out/p2_gemm final

out/naive_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p0.o out/tnns/ternarize2row$(ISA).o out/tnns/gemm-variants/mm_v1_p0_naive.o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v1_p0_naive.o out/tnns/ternarize2row$(ISA).o -o out/naive_gemm

out/p0_blocked_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v2_p0_blocked.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v2_p0_blocked.o out/tnns/ternarize2row$(ISA).o -o out/p0_blocked_gemm

out/p0_inst_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v3_p0_popcnt.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v3_p0_popcnt.o out/tnns/ternarize2row$(ISA).o -o out/p0_inst_gemm

out/p0_lib_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v4_p0_lib_2pc.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p0.o out/tnns/gemm-variants/mm_v4_p0_lib_2pc.o out/tnns/ternarize2row$(ISA).o -o out/p0_lib_gemm

out/p1_2pc_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v5_p1_lib_2pc.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v5_p1_lib_2pc.o out/tnns/ternarize2row$(ISA).o -o out/p1_2pc_gemm

out/p1_1pc_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v6_p1_lib_1pc.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v6_p1_lib_1pc.o out/tnns/ternarize2row$(ISA).o -o out/p1_1pc_gemm

out/p1_csas_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v7_p1_lib_csas.o out/tnns/ternarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p1.o out/tnns/gemm-variants/mm_v7_p1_lib_csas.o out/tnns/ternarize2row$(ISA).o -o out/p1_csas_gemm

out/p2_gemm: $(TIME_GEMM_SRC) $(TNNS_SRC) $(TNNS_HDR) out/tnns/tnns_cpu_mm_p2.o out/tnns/mm_p2_final_tbn$(ISA).o out/tnns/mm_p2_final_btn$(ISA).o out/tnns/mm_p2_final_tnn$(ISA).o out/tnns/mm_p2_final_bnn$(ISA).o out/tnns/ternarize2row$(ISA).o out/tnns/binarize2row$(ISA).o
	gcc $(CCFLAGS) $(TIME_GEMM_SRC) out/tnns/tnns_cpu_mm_p2.o out/tnns/mm_p2_final_bnn$(ISA).o out/tnns/mm_p2_final_tbn$(ISA).o out/tnns/mm_p2_final_btn$(ISA).o out/tnns/mm_p2_final_tnn$(ISA).o out/tnns/ternarize2row$(ISA).o out/tnns/binarize2row$(ISA).o -o out/p2_gemm
