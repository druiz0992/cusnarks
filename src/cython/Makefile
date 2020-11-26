# ------------------------------------------------------------------
#
#    Copyright 2018 0kims association.
#
#    This file is part of cusnarks.
#
#    cusnarks is a free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published by the
#    Free Software Foundation, either version 3 of the License, or (at your option)
#    any later version.
#
#    cusnarks is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
#    more details.
#
#    You should have received a copy of the GNU General Public License along with
#    cusnarks. If not, see <https://www.gnu.org/licenses/>.
#
#
# ------------------------------------------------------------------
# Author     : David Ruiz
#
# File name  : Makefile
#
# Date       : 03/02/2019
#
# ------------------------------------------------------------------
#


#####
# Define Paths

INCLUDE_PATH = $(INCLUDE_PATH)
OBJECT_PATH = $(OBJECT_PATH)
LIB_PATH = $(LIB_PATH)
CTSRC_PATH = $(CTSRC_PATH)
PYSRC_PATH = $(PYSRC_PATH)
LD_FLAGS=$(LD_FLAGS)

PYTHON=python3
#null :=
#WS:= $(null) $(null)
#DEFINE_MACROS:= $(subst $(WS),:,$(DEFINES))

setupScript=setup_nocuda.py
ifneq ($(FORCE_CPU),1)
	setupScript=setup.py
endif

build:
	$(PYTHON) ${setupScript} build -b $(OBJECT_PATH) build_ext -b $(LIB_PATH) --library-dirs=$(LIB_PATH)

clean:
	rm -f $(CTSRC_PATH)/*.cpp $(LIB_PATH)/*.so
	rm -rf $(OBJECT_PATH)

