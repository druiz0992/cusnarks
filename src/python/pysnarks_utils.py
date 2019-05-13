"""
/*
    Copyright 2018 0kims association.

    This file is part of cusnarks.

    cusnarks is a free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    cusnarks is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    cusnarks. If not, see <https://www.gnu.org/licenses/>.
*/

#  NOTES:
//
//
// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : pysnarks_utils.py
//
// Date       : 13/05/2019
//
// ------------------------------------------------------------------
//
// NOTES:

# 
# Python Snarks services utils implementation 
# 


// Description:
//    
//   TODO
//    
// ------------------------------------------------------------------

"""
import json,ast
import os.path
import numpy as np
import time

from random import randint

def json_to_dict(data, ):
    data = {str(k) : data[k] for k in data.keys()}
    for k, v in data.items():
        if type(data[k]) is list:
            json_to_list(data[k])
        elif type(data[k]) is dict:
            json_to_dict(data[k])
        elif is_long(v):
            data[k] = long(v)

    return data

def json_to_list(data):
     for idx,el in enumerate(data):
         if type(el) is list:
             json_to_list(el)
         elif type(el) is dict:
             data[idx] = json_to_dict(el)
         elif type(el) is unicode or type(el) is str:
             if el.isdigit():
               data[idx] = long(el)
             else:
                 data[idx] = el
     return

def is_long(input):
        try:
            num = long(input)
        except ValueError:
            return False
        return True

