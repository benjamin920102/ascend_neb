.PHONY: all reference build test-host test-npu clean
all: build
reference:
	bash scripts/setup_reference.sh
build:
	bash scripts/build.sh
test-host:
	bash tests/run_host_tests.sh

test-npu:
	./build/ascend_neb selftest --device 0 --rows 48 --attempts 3
clean:
	rm -rf build
