TARGET_ARCH = linux64

include Makefile.config

CXXFLAGS += -std=c++11
# LDFLAGS += -pg

PROJECT := kpn_yolo
SRCS    := kpn_yolo.cpp
OBJS    := $(SRCS:.cpp=.o)

include Makefile.rules
