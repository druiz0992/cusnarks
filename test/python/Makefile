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
# Date       : 31/01/2019
#
# ------------------------------------------------------------------

PYTHON=python3
#PYTHON_TESTS = $(shell find -name 'test_*.py')
PYTHON_TESTS = $(shell find -name 'test_cusnarks_*.py')

test:
	@for i in $(PYTHON_TESTS); do \
	echo "Starting Test :  $$i"; \
	$(PYTHON) $$i; \
	done

clean:
	rm -f *.pyc

build:
	echo "Nothing to do"
