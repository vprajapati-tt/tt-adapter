# "make" tt-mlir
cd third_party/tt-mlir
make
source env/activate
cd ../..

# Install tt-adapter into the current package
pip install -e .
