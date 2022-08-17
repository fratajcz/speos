#! /bin/env python

import signal, logging, sys, time

def do_computation_iteration():
    time.sleep(1)

# Global Boolean variable that indicates that a signal has been received
interrupted = False

# Global Boolean variable that indicates then natural end of the computations
converged = False

# Definition of the signal handler. All it does is flip the 'interrupted' variable
def signal_handler(signum, frame):
    print ('Terminate the process')
    # save results, whatever...
    sys.exit()

# Register the signal handler
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


while not interrupted and not converged:
    do_computation_iteration()    

# Save current state 
if interrupted:
    logging.error("Ended due to SIGTERM singal")
    print("Ended due to SIGTERM singal")
    sys.exit(99)
sys.exit(0)