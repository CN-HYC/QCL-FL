# 设置 Python 版本为 3.12
PYTHON_VERSION = 3.12

# 获取 Python 3.12 的 include 路径和链接标志
PYTHON_INCLUDE = $(shell python$(PYTHON_VERSION)-config --includes)
PYTHON_LIB = $(shell python$(PYTHON_VERSION)-config --ldflags)

# 目标文件
all: _example.so

# 生成共享库
_example.so: example.o example_wrap.o
	gcc -shared $^ -o $@ $(PYTHON_LIB)

# 编译 C 文件
example.o: example.c
	gcc -fPIC -c example.c -o $@

# 编译 SWIG 生成的 C++ 文件
example_wrap.o: example_wrap.cxx
	g++ -fPIC -c example_wrap.cxx -o $@ $(PYTHON_INCLUDE)

# 清理生成的文件
clean:
	rm -f *.o _example.so example_wrap.cxx example.py