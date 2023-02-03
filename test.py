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
l1 = [0,0,4,4,0,4,4,4]

l2 = [4,0,4,4,0,4,4,0]

def val_preprocessing(true, pred):
    new_true = []
    new_pred = []
    #if value in true and pred both = 0 remove do noting else add respective values to their respective lists
    for i in range(len(true)):
        if true[i] == 0 and pred[i] == 0:
            continue
        else:
            new_true.append(true[i])
            new_pred.append(pred[i])
    return new_true, new_pred

print(val_preprocessing(l1, l2))
            



    
        
            

            
    