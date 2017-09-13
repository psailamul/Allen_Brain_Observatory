from config import Allen_Brain_Observatory_Config
main_config=Allen_Brain_Observatory_Config()
import itertools

class RF_queries_generator():
    newid = itertools.count().next
    def __init__(self):
        self.id = RF_queries_generator.newid()
        self.queries ={
            'cell_specimen_id'=None , 
            'lsn_name'=None  ,  # 'locally_sparse_noise' 'locally_sparse_noise_eight_deg'
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
        conditions_dict ={
        'on_center_x':'[9, 50]',
        'on_center_y':'[9, 20)',
        'on_width_x':'[0.1,inf)',
        'on_width_y':'(1, inf)',
        'off_center_x':'(inf,50]',
        'off_center_y':'(9,20)',
        'off_width_x':'(2, 40]'
        }
        command_text=""
        FLAG_FIRST = False
        for attr, conditions in conditions_dict.iteritems(): # attr =  key of the thing you want
            if conditions.find(',') == -1 : # No seperator
                print "Error : unknown condition expression"
            else:

                if FLAG_FIRST:
                    command_text+=" and"
                else:
                    FLAG_FIRST = True
                lo_bound,up_bound = conditions.split(',') 
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





"""

            SELECT * FROM rf
            WHERE on_center_x >= %s and on_center_x < %s and on_center_y >= %s and on_center_y < %s

            %
            (namedict['x_min'], namedict['x_max'], namedict['y_min'], namedict['y_max']))
        bounds = conditions.split(',') # bounds[0] == lower bound , bounds[1] == upper bound
        

        if bounds[0].split('[')
        
        [ ]
        [ )
        ( ]
        ( )






test1 = "5, 10"
test2 = "[5.0,10]"
test3 = "[5.0,10)"
test4 = "[5.0 ,10.2 )"
test4 = "[5.0 , 10.2 )"
import re
pattern = re.compile("^([A-Z][0-9]+)+$")
pattern.match(string)



prog = re.compile(pattern)
result = prog.match(string)
result = re.match(pattern, string)

import re
re.findall("\d+\.\d+", "Current Level: 13.4 db.")
['13.4']

        regex = r"\(\f.,\f.\)"
        match = re.search(regex, '(4.5 , 5)') 
        


integers = []
floats = [] # Don't use float as a variable, it will override a built-in python function
not_number = []

# I modified this list so all the elements are string, if you already have ints and floats, you can use type() to know where to append
input_list = ["100", "234", 'random', "5.23", "55.55", 'random2']

for i in input_list:
    value = None
    try:
        value = int(i)
    except ValueError:
        try:
            value = float(i)
        except ValueError:
            not_number.append(i)
        else:
            floats.append(value)
    else:
        integers.append(value)

print(not_number)
print(floats)
print(integers)

# ['random', 'random2']
# [5.23, 55.55]
# [100, 234]

Test 
re.findall("\[.\d+.,|\[\d+.\d+,", "[5.0  , 60.0]")





    def specift_range_constraints(self, conditions):
        """
            Some math convention  for the conditional string
                '[' , ']' = close range
                '(' , ')' = open range
            Therefore 
                [A, B] : A <= x <= B
                [A, B) : A <= x <  B
                (A, B] : A <  x <= B
                (A, B) : A <  x <  B
        """
        if confitions.find(',') == -1 : # No seperator
            print "Error : unknown condition expression"
            return None
        else:
            lo_bound,up_bound = conditions.split(',') 
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
                cond = '>='
                uu=']'
            elif not up_bound.find(')')==-1:
                cond = '>'
                uu=')'
            else:
                print "Warning : No specify lower bound, use open range"
                cond = '<'
                uu=' '
            up_bound=up_bound.replace(uu,' ')
            upbound=float(up_bound)


        bounds = conditions.split(',') # bounds[0] == lower bound , bounds[1] == upper bound
        

        if bounds[0].split('[')
        
        [ ]
        [ )
        ( ]
        ( )




            def select_cells_by_rf_coor(self, namedict):
        """
        Select cells by rf coordinates.
        """
        self.cur.execute(
            """
            SELECT * FROM rf
            WHERE on_center_x >= %s and on_center_x < %s and on_center_y >= %s and on_center_y < %s
            """
            %
            (namedict['x_min'], namedict['x_max'], namedict['y_min'], namedict['y_max']))





test1 = "5, 10"
test2 = "[5.0,10]"
test3 = "[5.0,10)"
test4 = "[5.0 ,10.2 )"
test4 = "[5.0 , 10.2 )"
import re
pattern = re.compile("^([A-Z][0-9]+)+$")
pattern.match(string)



prog = re.compile(pattern)
result = prog.match(string)
result = re.match(pattern, string)

import re
re.findall("\d+\.\d+", "Current Level: 13.4 db.")
['13.4']

        regex = r"\(\f.,\f.\)"
        match = re.search(regex, '(4.5 , 5)') 
        


integers = []
floats = [] # Don't use float as a variable, it will override a built-in python function
not_number = []

# I modified this list so all the elements are string, if you already have ints and floats, you can use type() to know where to append
input_list = ["100", "234", 'random', "5.23", "55.55", 'random2']

for i in input_list:
    value = None
    try:
        value = int(i)
    except ValueError:
        try:
            value = float(i)
        except ValueError:
            not_number.append(i)
        else:
            floats.append(value)
    else:
        integers.append(value)

print(not_number)
print(floats)
print(integers)

# ['random', 'random2']
# [5.23, 55.55]
# [100, 234]

Test 
re.findall("\[.\d+.,|\[\d+.\d+,", "[5.0  , 60.0]")

"""