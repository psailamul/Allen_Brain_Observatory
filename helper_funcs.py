# All helper Functions for the project
import cPickle as pickle 

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inputfile: 
        return pickle.load( inputfile );
        
        
        
#Function to download cell response by experiment session ID (nwb file)

#Then read stim_table and reconstruct input matrix in full trials (ex. 5950 in NS)
#   Remove blank
#   Sort the response by frame/scene ID or not

#Have to support "filters" function ---> Cell of interest masking 


#IDEA: Build a function to download input/response by ID

