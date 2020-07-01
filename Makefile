all: mccubed datasketches

mccubed:
	@cd ./modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"

datasketches:
	@echo "Installing the datasketches package...\n"
	@cd modules/datasketches && python setup.py build && pip install .
	@echo "pip may have complained above about the build failing, but it may "
	@echo "have\nstill been successfully installed.  To test, try to import "
	@echo "datasketches in a\nPython session.\n"

