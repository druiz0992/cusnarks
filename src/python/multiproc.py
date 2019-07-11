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

// ------------------------------------------------------------------
// Author     : David Ruiz
//
// File name  : muliproc.py
//
// Date       : 03/07/2019
//
// ------------------------------------------------------------------
//
// Description:
//  Implements parallel processing
//
// ------------------------------------------------------------------

"""

import numpy as np
import multiprocessing as mp

class MP(object):

    def __init__(self, max_workers=None):
        """
          Constructor
        """
        if max_workers is None:
           self.nworkers = mp.cpu_count()-1
        else:
           self.nworkers = min(mp.cpu_count()-1, max_workers)

        self.n_active_workers = 0
        self.worker = mp.Pool(processes = self.nworkers)
        self.status = np.zeros(self.nworkers,dtype=np.uint32)
        self.results = np.asarray(self.nworkers * [None])
        

    def _get_idle(self,count, reserve=False):
        c = 0
        idx = []
        if reserve:
          for i, v in enumerate(self.status==0):
            if v:
              if c == count:
                break
              self.status[i] = 1
              idx.append(i)
              c+=1
            
          self.n_active_workers = np.count_nonzero(self.status==1)

        return np.asarray(idx, dtype=np.uint32)

    def _clear(self):
       return

    def launch(self, count, func, args):
        if self.nworkers < self.n_active_workers + count:
           return -1

        idx = self._get_idle(count, reserve=True)

        self.results[idx] = self.worker.apply_async(self.f, args=args, callback=self._clear)

    def get_results(self, w=None):
        if w is None:
            results = [r.get() for r in self.results if r is not None] 
            self.results = np.asarray(self.nworkers * [None])
            return results
       
