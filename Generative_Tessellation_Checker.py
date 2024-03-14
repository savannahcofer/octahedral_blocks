#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:41:29 2024

@author: savannahcofer
"""
# a is the waterbomb point in the x direction
# b is the waterbomb point in the y direction
# c is the waterbomb point in the z direction
# n is no cell in this spot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for all the dimensions of x_len, y_len, and z_len, propegate all a b and c
x_len = 2
y_len = 3
z_len = 2
unit = np.zeros((x_len, y_len, z_len), dtype=object)
unit[0][0][0] = 'c'
# Set all values in unit to 'n'
unit[:] = 'n'

# Input a unit cell (3D list array containing a, b, c)
# of dimensions x_len, y_len, and z_len
# Output is whether it is a valid tessellation
def check_if_valid(unit, x_len, y_len, z_len):
    val = True
    # check if valid across all x lines
    for z in range(z_len):
        
        #print("z is",z)
        for y in range(y_len):
            #print("z is",z,"y is",y)
            for x in range(x_len):
                #print("z is",z,"y is",y,"x is",x)
                if x == (x_len - 1): # if we are at the end of the line, check the first
                    val = truth(unit[x][y][z],unit[0][y][z],'x')
                else: # we check the next one down the line
                    val = truth(unit[x][y][z],unit[x+1][y][z],'x')
                
                if not val:
                    return False # return immediately if false is found
            # end for x
        # end for y
    # end for z
    # check if valid across all y lines
    for z in range(z_len):
        #print("z is",z)
        for x in range(x_len):
            #print("z is",z,"x is",x)
            for y in range(y_len):
                #print("z is",z,"x is",x,"y is",y)
                if y == (y_len - 1): # if we are at the end of the line, check the first
                    val = truth(unit[x][y][z],unit[x][0][z],'y')
                else: # we check the next one down the line
                    val = truth(unit[x][y][z],unit[x][y+1][z],'y')
                
                if not val:
                    return False # return immediately if false is found
                # end for y
        # end for x
    # end for z
    # check if valid across all y lines
    for x in range(x_len):
        #print("x is",x)
        for y in range(y_len):
            #print("x is",x,"y is",y)
            for z in range(z_len):
                #print("x is",x,"y is",y,"z is",z)
                if z == (z_len - 1): # if we are at the end of the line, check the first
                    val = truth(unit[x][y][z],unit[x][y][0],'z')
                else: # we check the next one down the line
                    val = truth(unit[x][y][z],unit[x][y][z+1],'z')
                
                if not val:
                    return False # return immediately if false is found
                # end for z
        # end for y
    # end for x
    return val
# end check_if_valid

# Input is two single units of the unit cell which can be 'a', 'b', 'c', or 'n'
# Dimension is what we are comparing as adjacent to each other
def truth(unit1, unit2, dim):
    result = False
    if unit1 == unit2:
        return True
    elif unit1 == 'n' or unit2 == 'n':
        return True
    else:
        if dim == 'x':
            if (unit1 != 'b' and unit2 != 'b'):
                return True
        elif dim == 'y':
            if (unit1 != 'c' and unit2 != 'c'):
                return True
        elif dim == 'z':
            if (unit1 != 'a' and unit2 != 'a'):
                return True
            
    return result

check = check_if_valid(unit, x_len, y_len, z_len)
#result = truth('a','b','y')
#print(result)
#print(unit[0][0][0])

# Input is the unit cell size x_len, y_len, and z_len
# Output is all the cells that are valid tessellations
def permutation_generator(x_len, y_len, z_len):
    new_unit = np.zeros((x_len, y_len, z_len), dtype=object)
    i = 0
    iteration = 0
    # Set all values in new_unit to 'n'
    new_unit[:] = 'n'
    # Add the new unit and its boolean value to units_dict
    df_units = pd.DataFrame()
    df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}", 'Array': new_unit, 'IsValid': True, 'Iteration': iteration}])], ignore_index=True)   
    
    # we run through each x, y, z
    for z in range(z_len):
        for y in range(y_len):
            for x in range(x_len):
                # each new xyz space we add a valid option of our previous true options
                prev_df_units = df_units.loc[(df_units['IsValid'] == True) &
                                             (df_units['Iteration'] == iteration)]
                iteration = iteration + 1
                print(x,"+" ,y,"+",z)
                for __, row in prev_df_units.iterrows():
                    new_unit = row['Array'].copy()
                    new_unit[x][y][z] = 'a'
                    i = i+1
                    if check_if_valid(new_unit, x_len, y_len, z_len): # if true, keep
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': True, 'Iteration': iteration}])], ignore_index=True)   
                    else:
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': False, 'Iteration': iteration}])], ignore_index=True)
                    
                    new_unit[x][y][z] = 'b'
                    i = i+1
                    if check_if_valid(new_unit, x_len, y_len, z_len): # if true, keep
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': True, 'Iteration': iteration}])], ignore_index=True)
                    else:
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': False, 'Iteration': iteration}])], ignore_index=True)
                        
                    new_unit[x][y][z] = 'c'
                    i = i+1
                    if check_if_valid(new_unit, x_len, y_len, z_len): # if true, keep
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': True, 'Iteration': iteration}])], ignore_index=True)
                    else:
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': False, 'Iteration': iteration}])], ignore_index=True)
                        
                    new_unit[x][y][z] = 'n'
                    i = i+1
                    if check_if_valid(new_unit, x_len, y_len, z_len): # if true, keep
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': True, 'Iteration': iteration}])], ignore_index=True)
                    else:
                        df_units = pd.concat([df_units, pd.DataFrame([{'UnitID': f"unit{i}",
                                                                        'Array': new_unit.copy(), 'IsValid': False, 'Iteration': iteration}])], ignore_index=True)
                    
    # finally, get rid of all the 'n' disconnected units                

    return df_units

#df_units = pd.DataFrame()
#df_units = permutation_generator(2, 2, 2)


#%% See if array is connected

# Depth-first search (DFS) algorithm

def is_connected(array, x_len, y_len, z_len):
    def dfs(x, y, z):
        if x < 0 or x >= len(array) or y < 0 or y >= len(array[0]) or z < 0 or z >= len(array[0][0]) or not array[x][y][z] or visited[x][y][z]:
            return
        visited[x][y][z] = True
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        for dx, dy, dz in directions:
            dfs(x+dx, y+dy, z+dz)

    # end dfs
    
    # create np array of false values
    visited = np.full((x_len, y_len, z_len), False)
    start = None
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                if array[x][y][z] != 'n':
                    start = (x, y, z)
                    #print('start point is,', start)
                    break
            if start:
                break
        if start:
            break

    if not start:
        #print('empty array')
        return False  # Empty array is considered fully connected

    dfs(start[0], start[1], start[2])

    for x in range(len(array)):
        for y in range(len(array[0])):
            for z in range(len(array[0][0])):
                if array[x][y][z] and not visited[x][y][z]:
                    return False
    
    # look for the case if there is a whole slice of null values
    # Check for a whole null slice in the x direction
    for x in range(x_len):
        if np.all(array[x, :, :] == 'n'):
            return False

    # Check for a whole null slice in the y direction
    for y in range(y_len):
        if np.all(array[:, y, :] == 'n'):
            return False

    # Check for a whole null slice in the z direction
    for z in range(z_len):
        if np.all(array[:, :, z] == 'n'):
            return False
        
    # otherwise, return true!
    return True

#array = np.array([[['a', 'a'], ['n', 'n']], [['a', 'a'], ['n', 'n']]])
#print(is_connected(array, 2, 2, 2))

#%% Remove low DOF pieces
# Returns dof_array, an array of connectivity
def compute_DOF(array, x_len, y_len, z_len):
    # create np array of 0 values
    dof_array = np.full((x_len, y_len, z_len), 0)
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                dof = 0;
                #print('ARRAY VALUE IS', array[x][y][z])
                if array[x][y][z] == 'n':
                    dof = dof + 99 # a large random number of dof
                elif array[x][y][z] != 'n':
                    # our current cell is a
                    if array[x][y][z] == 'a':
                        # in the positive x direction
                        if array[(x+1) % x_len][y][z] == 'a':
                            dof = dof + 2
                        elif array[(x+1) % x_len][y][z] == 'c':
                            dof = dof + 4
                        # in the negative x direction
                        if array[(x-1) % x_len][y][z] == 'a':   
                            dof = dof + 2
                        elif array[(x-1) % x_len][y][z] == 'c':
                            dof = dof + 4 
                        # in the positive y direction
                        if array[x][(y+1) % y_len][z] == 'a':
                            dof = dof + 2
                        elif array[x][(y+1) % y_len][z] == 'b':
                            dof = dof + 4   
                        # in the negative y direction
                        if array[x][(y-1) % y_len][z] == 'a':
                            dof = dof + 2
                        elif array[x][(y-1) % y_len][z] == 'b':
                            dof = dof + 4 
                        # in the positive z direction
                        if array[x][y][(z+1) % z_len] == 'a':
                            dof = dof + 1
                        # in the negative z direction
                        if array[x][y][(z-1) % z_len] == 'a':
                            dof = dof + 1
                    # our current cell is b
                    if array[x][y][z] == 'b':
                        # in the positive x direction
                        if array[(x+1) % x_len][y][z] == 'b':
                            dof = dof + 1
                        # in the negative x direction
                        if array[(x-1) % x_len][y][z] == 'b':   
                            dof = dof + 1
                        # in the positive y direction
                        if array[x][(y+1) % y_len][z] == 'a':
                            dof = dof + 4
                        elif array[x][(y+1) % y_len][z] == 'b':
                            dof = dof + 2
                        # in the negative y direction
                        if array[x][(y-1) % y_len][z] == 'a':
                            dof = dof + 4
                        elif array[x][(y-1) % y_len][z] == 'b':
                            dof = dof + 2 
                        # in the positive z direction
                        if array[x][y][(z+1) % z_len] == 'b':
                            dof = dof + 2
                        elif array[x][y][(z+1) % z_len] == 'c':
                            dof = dof + 4  
                        # in the negative z direction
                        if array[x][y][(z-1) % z_len] == 'b':
                            dof = dof + 2
                        elif array[x][y][(z-1) % z_len] == 'c':
                            dof = dof + 4  
                    # our current cell is c
                    if array[x][y][z] == 'c':
                        # in the positive x direction
                        if array[(x+1) % x_len][y][z] == 'a':
                            dof = dof + 4
                        elif array[(x+1) % x_len][y][z] == 'c':
                            dof = dof + 2    
                        # in the negative x direction
                        if array[(x-1) % x_len][y][z] == 'a':   
                            dof = dof + 4
                        if array[(x-1) % x_len][y][z] == 'c':   
                            dof = dof + 2    
                        # in the positive y direction
                        if array[x][(y+1) % y_len][z] == 'c':
                            dof = dof + 1
                        # in the negative y direction
                        if array[x][(y-1) % y_len][z] == 'c':
                            dof = dof + 1
                        # in the positive z direction
                        if array[x][y][(z+1) % z_len] == 'b':
                            dof = dof + 4
                        elif array[x][y][(z+1) % z_len] == 'c':
                            dof = dof + 2 
                        # in the negative z direction
                        if array[x][y][(z-1) % z_len] == 'b':
                            dof = dof + 4
                        elif array[x][y][(z-1) % z_len] == 'c':
                            dof = dof + 2 
                    # end if
                    
                dof_array[x][y][z] = dof
    # end iterating through all array values
    return dof_array

#%% Remove arrays with low dof
# Pass in the connected df_units_conn
def remove_low_DOF(df_units_conn, x_len, y_len, z_len):
    # Initialize an empty DataFrame for the final units
    df_units_final = pd.DataFrame(columns=['UnitID', 'Array', 'IsValid', 'Iteration'])
    
    # Iterate through each unit in the connected DataFrame
    for index, row in df_units_conn.iterrows():
        # Compute the Degree of Freedom (DOF) array for the current unit
        dof_array = compute_DOF(row['Array'], x_len, y_len, z_len)
        #print('Printing!')
        #print(row['Array'])
        #print(dof_array)
        # Check if all values in the DOF array are greater than or equal to 3
        if np.all(dof_array >= 3):
            # If true, append the current unit to the final DataFrame
            df_units_final = pd.concat([df_units_final, pd.DataFrame([row])], ignore_index=True, axis=0)
    
    return df_units_final


#%% Test the arrays
# 1x1x1
# pruned = 4
# connected = 3
# final = 3
# All have 10 connnections
x_len = 4
y_len = 4
z_len = 4
print('running for',x_len,y_len,z_len)
df_units = permutation_generator(x_len, y_len, z_len);
max_iteration = df_units['Iteration'].max()
df_units_pruned = df_units[(df_units['Iteration'] == max_iteration) & (df_units['IsValid'] == True)]
print('generated units')
#df_units_pruned_copy = df_units_pruned.copy()
#df_units_pruned_copy['Array_Str'] = df_units_pruned_copy['Array'].apply(lambda x: ''.join(x.flatten()))

#df_units_unique = df_units_pruned_copy.drop_duplicates(subset=['Array_Str'])
#df_units_unique = df_units_unique.drop(columns=['Array_Str'])

#print('dropped duplicates')

df_units_conn = pd.DataFrame(columns=['UnitID', 'Array', 'IsValid', 'Iteration'])

for index, row in df_units_pruned.iterrows():
    if is_connected(row['Array'], x_len, y_len, z_len):
        df_units_conn = pd.concat([df_units_conn, pd.DataFrame([row])], ignore_index=True, axis=0)

df_units_final = remove_low_DOF(df_units_conn, x_len, y_len, z_len)
print('dropped disconnected')
# Sample data
categories = ['a.', 'b.', 'c.', 'd.']
values = np.array([0, 0, 0, 0])
values[0] = 4**(x_len*y_len*z_len)
values[1] = len(df_units_pruned)
values[2] = len(df_units_conn)
values[3] = len(df_units_final)

fig, ax = plt.subplots()

# Create bar plot
colors = ['#202122', '#908876', '#dad5c7', '#e6ddc9']
plt.bar(categories, values, color=colors)
plt.tight_layout()
# Add title and labels
plt.ylabel('# Perm (Log Scale)', fontsize=28)
plt.tick_params(axis='y', labelsize=20)
plt.tick_params(axis='x', labelsize=20)
# Set y-axis to logarithmic scale
plt.yscale('log')
plt.ylim(bottom=1)

# print total number
num_total = len(df_units_final)
print('num total is', num_total)

# Show plot
plt.gca().figure.patch.set_edgecolor('none')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{num_total}_4x4x4', bbox_inches='tight')
plt.show()






