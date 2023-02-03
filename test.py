'''i have two lists 

l1 = [0,0,4,4,0,4,4]

l2 = [0,0,4,0,0,4,4]

write the following python script:

- iter over l1
- create a empty list "l_index"  and "l_items" and "result"
- check if item is != 0 if yes save index in l_index and go to next element. if elem same as before add element and index of it to respective lists. repeat as long its the case 
- once previous is not the case compare the numbers from the first number in l_index to the last number in l_index from l2 with the numbers in l_items. if they are the same add the value of the number else 0
- repeat as long as the list goes 

In the end the result should be 

result = [0, 4]
'''
l1 = [0,0,4,4,0,4,4,0]

l2 = [0,0,4,4,0,4,4,0]

'''def process(l1, l2):
    l_index =[]
    l_items = []
    result = [] 
    for i, elem in enumerate(l1):
        if elem != 0 and l_items == [] or elem != 0 and elem == l_items[-1]:
            l_index.append(i)
            l_items.append(elem)
        else:
            if elem == 0 and l_index == []:
                pass
            elif l2[l_index[0]:l_index[-1]+1] == l_items:
                #sollte 
                #print(l_items)
                result.append(l_items[0])
            else:
                result.append(0)
            l_index = []
            l_items = []
    if l_index != []:
        if l2[l_index[0]:l_index[-1]+1] == l_items:
            result.append(elem)
        else:
            result.append(0)
    return result
            
print(process(l1, l2))'''

def process(l1, l2):
    #drop all items where 
    l_index =[]
    l_items = []
    new_true = [] 
    new_pred = []
    for i, elem in enumerate(l1):
    
#process(l1, l2)



    
        
            

            
    