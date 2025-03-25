#
# Makefile to call protocol buffer compiler for Python
#

CC          :=protoc
SRC_DIR 	:=proto/
SRC         :=$(wildcard $(SRC_DIR)*.proto)

PYTHON_OUT_DIR :=modules/interfaces
PYTHON_OUT     :=$(patsubst $(SRC_DIR)%.proto,$(PYTHON_OUT_DIR)/%_pb2.py,$(SRC))

all: $(PYTHON_OUT)

$(PYTHON_OUT): $(SRC)
	$(CC) --proto_path=$(SRC_DIR) $(SRC) --python_out=$(PYTHON_OUT_DIR)

clean:
	rm -f $(PYTHON_OUT_DIR)/*.py
