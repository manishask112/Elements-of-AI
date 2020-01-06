#!/usr/local/bin/python3
#
#Code by: [Manisha Suresh Kumar(msureshk), Uma Maheswari Gollapudi(ugollap)]
from queue import PriorityQueue
import sys

REVISITED_start=[]
REVISITED_end=[] 

#To find a final path from the bidirectional fringes
def final_path(start_path,end_path):
    end_path.reverse()
    final_path=[]
    for i in range(0,len(start_path)):
        final_path.append(start_path[i])
        if start_path[i] in end_path:
            j=end_path.index(start_path[i])
            remaining_path=end_path[j+1:]
            final_path.extend(remaining_path)
            break 

    return calculate_parameters(final_path)

#Calculate segments,distance, time and total gas gallons of the final path
def calculate_parameters(final_path): 
    distance=0
    time=float(0)
    mpg=float(0)
    for i in range(0,len(final_path)-1):
        for j in range(0,len(road_segments[final_path[i]])):
            if road_segments[final_path[i]][j][0]==final_path[i+1]:
                distance=distance+int(road_segments[final_path[i]][j][1])
                time=time+(float(road_segments[final_path[i]][j][1])/float(road_segments[final_path[i]][j][2]))
                mpg=mpg+((400*(float(road_segments[final_path[i]][j][2])/150))*(1-(float(road_segments[final_path[i]][j][1])/150))**4)
         
    return (len(final_path)-1,(distance,time,mpg),final_path)          

#To find path for cost function segments using bi-directional strategy
def solve_segments(start_city,end_city):
    fringe_start=PriorityQueue()
    fringe_end=PriorityQueue()
    fringe_start.put((0,start_city,[start_city]))
    fringe_end.put((0,end_city,[end_city]))
    while not fringe_start.empty() or not fringe_end.empty():
        (segments_moved_start,curr_city_start,route_so_far_start)=fringe_start.get()
        (segments_moved_end,curr_city_end,route_so_far_end)=fringe_end.get()
        for i in range(0,len(fringe_end.queue)):
            if(curr_city_start in fringe_end.queue[i][2]):
                return final_path(route_so_far_start,fringe_end.queue[i][2])
        for i in range(0,len(fringe_start.queue)):
            if(curr_city_end in fringe_start.queue[i][2]):
                return final_path(fringe_start.queue[i][2],route_so_far_end)
        REVISITED_start.append(curr_city_start)
        REVISITED_end.append(curr_city_end)
        for successors in explore(curr_city_start,REVISITED_start):
            add_next_route=route_so_far_start[:]
            add_next_route.append(successors[0])
            fringe_start.put((segments_moved_start+1,successors[0],add_next_route))
        for successors in explore(curr_city_end,REVISITED_end):    
            add_next_route=route_so_far_end[:]
            add_next_route.append(successors[0])
            fringe_end.put((segments_moved_end+1,successors[0],add_next_route))       
    return False

#To find path for cost function distance
def solve_distance(start_city,end_city):
    fringe=PriorityQueue()
    fringe.put((0,(0,float(0),float(0)),start_city,[start_city]))
    while not fringe.empty():
        (distance_travelled,parameters,curr_city,route_so_far)=fringe.get()
        if curr_city==end_city:
            return (len(route_so_far)-1,parameters,route_so_far)
        REVISITED_start.append(curr_city)
        for successors in explore(curr_city,REVISITED_start):
            add_next_route=route_so_far[:]
            add_next_route.append(successors[0])
            fringe.put((distance_travelled+successors[1],(parameters[0]+successors[1],parameters[1]+successors[2],parameters[2]+successors[3]),successors[0],add_next_route))
    return False

#To find path for cost function time
def solve_time(start_city,end_city):
    fringe=PriorityQueue()
    fringe.put((float(0),(0,float(0),float(0)),start_city,[start_city]))
    while not fringe.empty():
        (time_travelled,parameters,curr_city,route_so_far)=fringe.get() 
        if curr_city==end_city:
            return (len(route_so_far)-1,parameters,route_so_far)
        REVISITED_start.append(curr_city)
        for successors in explore(curr_city,REVISITED_start):
            add_next_route=route_so_far[:]
            add_next_route.append(successors[0])
            fringe.put((time_travelled+successors[2],(parameters[0]+successors[1],parameters[1]+successors[2],parameters[2]+successors[3]),successors[0],add_next_route))
    return False

#To find path for cost function mpg
def solve_mpg(start_city,end_city):
    fringe=PriorityQueue()
    fringe.put((float(0),(0,float(0),float(0)),start_city,[start_city]))
    while not fringe.empty():
        (mpg_so_far,parameters,curr_city,route_so_far)=fringe.get()
        if curr_city==end_city:
            return (len(route_so_far)-1,parameters,route_so_far)
        REVISITED_start.append(curr_city)
        for successors in explore(curr_city,REVISITED_start):
            add_next_route=route_so_far[:]
            add_next_route.append(successors[0])
            fringe.put((mpg_so_far+successors[3],(parameters[0]+successors[1],parameters[1]+successors[2],parameters[2]+successors[3]),successors[0],add_next_route))
    return False

#Exploring children of current node and calculating its distance, time and mpg
def explore(curr_city,REVISITED):
    successors=[]
    for i in road_segments[curr_city]:
        if i[0] not in REVISITED:
            mpg=(400*(float(i[2])/150))*(1-(float(i[2])/150))**4
            successors.append((i[0],int(i[1]),float(i[1])/float(i[2]),mpg))     
            #                  city,distance,          time,          mpg
    return successors    

if __name__ == "__main__":
    city_gps = {}
    with open("city-gps.txt", 'r') as file:
        for line in file:
            city=[]
            for i in line.split():
                city += [i]
            city_gps.update({city[0]:(city[1],city[2])})    
    road_segments={}
    
    with open("road-segments.txt", 'r') as file:
        for line in file:
            highway=[]
            for i in line.split():
                highway+=[i]
            if highway[0] in road_segments:
                road_segments[highway[0]].append(highway[1:len(highway)])    
            else:
                road_segments.update({highway[0]:[highway[1:len(highway)]]})
            if highway[1] not in road_segments:
                road_segments.update({highway[1]:[[highway[0],highway[2],highway[3],highway[4]]]})
            else:
                road_segments[highway[1]].append([highway[0],highway[2],highway[3],highway[4]])
    start_city=sys.argv[1]
    end_city=sys.argv[2]
    if(sys.argv[3]=="segments"):
        answer=solve_segments(start_city,end_city)
    if(sys.argv[3]=="distance"):
        answer=solve_distance(start_city,end_city)    
    if(sys.argv[3]=="time"):
        answer=solve_time(start_city,end_city)   
    if(sys.argv[3]=="mpg"):
        answer=solve_mpg(start_city,end_city)     
    if answer==False:
        print("Inf")
    else:    
        print("The path to be taken:")
        for i in range(0,len(answer[2])-1):
            print(answer[2][i]+"-->",end='')
        print(answer[2][len(answer[2])-1])   
        print("Number of turns: "+str(answer[0]))
        print("Total distance to travel: "+str(answer[1][0])+" miles")
        print("Total duration of journey: "+str(answer[1][1])+" hr(s)")
        print("Total gas gallons: "+str(float(answer[1][0])/float(answer[1][2])))
        print(str(answer[0])+" "+str(answer[1][0])+" "+str(answer[1][1])+" "+str(float(answer[1][0])/float(answer[1][2])),end=' ') 
        for i in range(0,len(answer[2])-1):
                print(answer[2][i],end=' ')
        print(answer[2][len(answer[2])-1])         
