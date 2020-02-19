
mccubed:
	@if [ ! -d "./modules/MCcubed" ]; then                                    \
		echo "\nCloning MCcubed...";                                          \
		git clone https://github.com/pcubillos/mc3 modules/MCcubed/;          \
		echo "Finished cloning MCcubed into 'modules'.\n";                    \
		echo "Switching to the compatible MCcubed version...";                \
		cd modules/MCcubed;                                                   \
		git checkout mpi;                                                     \
		cd ../..;                                                             \
	else                                                                      \
		echo "MCcubed already exists.\n";                                     \
	fi
	@echo "\nModifying files within MCcubed..."
	@yes | cp -R code/MCcubed/ modules/
	@echo "\nCompiling MCcubed..."
	@cd ./modules/MCcubed && make
	@echo "Finished compiling MCcubed.\n"
