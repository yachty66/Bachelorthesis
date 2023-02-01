true = [0,0,0,4,5,0,0,0,0,0,0,8,0,0,0,0,0]
pred = [0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0]



new_pred = []
for i in range(len(true)):
    if true[i] != 0:
        new_pred.append(pred[i])
        
        
print(new_pred)

new_true = [i for i in true if i != 0]

print(new_true)