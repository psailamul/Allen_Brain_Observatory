from config import Allen_Brain_Observatory_Config
main_config=Allen_Brain_Observatory_Config()
import itertools
"""
Class for generate RF queries by any columns
Steps 
1. Create a setting_dict for the desired parameters  *more on this later
2. Q = RF_queries_generator()
3. Q.specify_search_constraint(setting_dicts)
4. after finish all setting run 
    COMMAND_STRING = q.finalized()
5. send the COMMAND_STRING to gather queries from db_main


*Useful setting 
    For range variable (ex center_x, center_y, width etc )
    specify a list of lower bound and upper bound 
    ex.  to get both on_center_x and on_center_y from 9 - 50 
        setting_dicts['on_center_x'] = [9,50]
        setting_dicts['on_center_y'] = [9,50]

"""
class RF_queries_generator():
    newid = itertools.count().next
    def __init__(self):
        self.id = RF_queries_generator.newid()
        self.queries ={
            'cell_specimen_id'=None , 
            'lsn_name'=None  , 
            'experiment_container_id'=None  , 
            'found_on'=None  , 
            'found_off'=None  ,  
            'alpha'=None, 
            'number_of_shuffles'=None , # non range
            'on_distance'=None , 
            'on_area'=None ,
            'on_overlap'=None ,
            'on_height'=None ,
            'on_center_x'=None ,
            'on_center_y'=None ,
            'on_width_x'=None ,
            'on_width_y'=None ,
            'on_rotation'=None , 
            'off_distance'=None , 
            'off_area'=None , 
            'off_overlap'=None ,
            'off_height'=None , 
            'off_center_x'=None , 
            'off_center_y'=None , 
            'off_width_x'=None ,
            'off_width_y'=None , 
            'off_rotation'=None , 
            }
        self.char_columns =['cell_specimen_id', 'lsn_name', 'experiment_container_id'] # note : treat cell id as string
        self.boolean_columns=['found_on','found_off']

    def specify_search_constraint(self, setting_dicts):
        """ setting values to a specific column directly. 
            EX setting_dicts = {'found_on':True, 'found_off':True} will set both queries['found_on'] and queries['found_off']=True
        """
        for k,v in setting_dicts.iteritems():
            self.queries[k]=v

    def finalized(self):
        """ SELECT * FROM rf where """

        COMMAND_STRING = "SELECT * FROM rf"
        FLAG_FIRST = False
        for k,v in self.queries.iteritems():
            if FLAG_FIRST:
                COMMAND_STRING += " and"
            if v is not None:
                if not FLAG_FIRST:
                    COMMAND_STRING+=" where"
                    FLAG_FIRST=True
                if k in self.char_columns or k in self.boolean_columns:
                    COMMAND_STRING+= " %s=%s"%(k,v)
                else:
                    if isinstance(v, int) or isinstance(v, float) : # get unique value ex. alpha=0.5
                        COMMAND_STRING+= " %s=%s"%(k,v)
                    elif instance(v, list):
                        COMMAND_STRING+= command_from_list_of_bounds(k,v)
                    else:
                        COMMAND_STRING+= get_range_constraints({k:v})
        return COMMAND_STRING  # Note : Have to make a function in db/db_main.py to run command from command string

    def command_from_list_of_bounds(key,list_of_bounds):
        """ 
            [A, B] A <= x <= B  
        """
        if len(list_of_bounds) ==2:
            A = list_of_bounds[0]
            B = list_of_bounds[1]
            command_text = " %s >= %s and %s <= %s"%(key,A,key,B)
            return command_text
        else:
            command_text=" %s=%s"%(key, list_of_bounds[0])
            for i in np.arange(1,len(list_of_bounds)):
                command_text+=" and %s=%s"%(key, list_of_bounds[i]) 
            return command_text


    def get_range_constraints(conditions_dict):
        """
            Some math convention  for the conditional string
                '[' , ']' = close range
                '(' , ')' = open range
                'inf' for unbound
            Therefore 
                [A, B] : A <= x <= B
                [A, B) : A <= x <  B
                (A, B] : A <  x <= B
                (A, B) : A <  x <  B
                (None, B) : x < B
                [A, inf) : x >= A
        """
        """
        conditions_dict ={
        'on_center_x':'[9, 50]',
        'on_center_y':'[9, 20)',
        'on_width_x':'[0.1,inf)',
        'on_width_y':'(1, inf)',
        'off_center_x':'(inf,50]',
        'off_center_y':'(9,20)',
        'off_width_x':'(2, 40]'
        } # For testing
        """
        # Note : There're bugs in here. Don't use it yet
        command_text=""
        FLAG_FIRST = False
        for attr, conditions in conditions_dict.iteritems(): # attr =  key of the thing you want
            if conditions.find(',') == -1 : # No seperator
                print "Error : unknown condition expression"
            else:
                if FLAG_FIRST:
                    command_text+= " and"
                else:
                    FLAG_FIRST = True
                lo_bound,up_bound = conditions.split(',') 

                if lo_bound.find('[') == -1:
                    
                if not lo_bound.find('[') == -1:
                    cond = '>='
                    ll='['
                elif not lo_bound.find('(')==-1:
                    cond = '>'
                    ll='('
                else:
                    print "Warning : No specify lower bound, use open range"
                    cond = '>='
                    ll=' '
                lo_bound=lo_bound.replace(ll,' ')
                lobound=float(lo_bound)

                if not up_bound.find(']') == -1:
                    cond = '<='
                    uu=']'
                elif not up_bound.find(')')==-1:
                    cond = '<'
                    uu=')'
                else:
                    print "Warning : No specify lower bound, use open range"
                    cond = '<'
                    uu=' '
                up_bound=up_bound.replace(uu,' ')
                upbound=float(up_bound)
                FOUND_LO = False
                if not(lobound ==float('inf') or lobound ==float('-inf')):
                    FOUND_LO = True
                    command_text += " %s %s %s"%(attr,cond,lobound)
                if not upbound ==float('inf'):
                    if FOUND_LO:
                        command_text +=" and"
                    command_text += " %s %s %s"%(attr,cond,upbound)
        return command_text

