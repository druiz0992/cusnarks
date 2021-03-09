import logging
import logging.handlers as handlers
from datetime import datetime



class CuLogger(object):
    def __init__(self, logLevel, logf):
        self.logger = None
        self.logLevel=""

        # Logger setup
        if logf is not None :
          self.type=0
          self.logger = logging.getLogger('cusnarks')

          formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

          ### Create new log file every day. Keep latest 7
          logHandler = handlers.TimedRotatingFileHandler(
                                logf,
                                when='D',
                                interval=1,
                                backupCount=7)
          logHandler.setLevel(logLevel)
          logHandler.setFormatter(formatter)
          self.logger.addHandler(logHandler)
          self.logger.setLevel(logLevel)
        else:
          self.type=1
          if logLevel == logging.INFO:
              self.logLevel = "INFO"
          else:
              self.logLevel = "ERROR"

    def info(self,*args):
        if self.type == 0:
            self.logger.info(" ".join(map(str,args)))
        else:
            now = datetime.now().strftime("%Y-%d-%m %H:%M:%S.%f")
            print( "[CUSNARKS] - " + now + " - " + self.logLevel + " - " +  " ".join(map(str,args)))

    def error(self,*args):
        if self.type == 0:
            self.logger.error(" ".join(map(str,args)))
        else:
            now = datetime.now().strftime("%Y-%d-%m %H:%M:%S.%f")
            print( "[CUSNARKS] - " + now + " - " + self.logLevel + " - " +  " ".join(map(str,args)))
