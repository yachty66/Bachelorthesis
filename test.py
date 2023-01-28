'''
- [x] from incoming create from each list item a list with key value pairs and if list item empty create an empty list
    - [x] handle empty strings
- [x] check if len of new income and actual is the same 
- [x] iter over new incoming and actual - iter over new actual and check if it appears in new incoming if yes add to list the key of the value else 0
- [ ] find a solution to the problem of having smaller case letters with upper case ones mixed up
    - [x] find out if its okay to turn all uppercase letters into smallercase letters in my incoming
- [x] get two the same examples like in true but for pred
- [x] test what happens with just an empty string 
- [x] add support for lists
- [x] add support for all of my attributes
- [ ] adding everything into one method and test it
- [ ] next steps 

# Notes
I think its not okay because in my token classification i am also not doing it  - in the end it doesnt matter. it does matter because it learns the whole representation
i am not sure yet I should keep that in mind. 
'''
import re

def label_pred(incoming, actual):
    l_targets = [[tuple_list[0] for tuple_list in sublist] for sublist in actual]
    l_predictions = []
    for string in incoming:
        matches = [match for match in re.findall(r"(rna: (.+?))(;|$)|(dna: (.+?))(;|$)|(cell_line: (.+?))(;|$)|(protein: (.+?))(;|$)|(cell_type: (.+?))(;|$)", string) if match[1] or match[4] or match[7] or match[10] or match[13]]
        inner_list = []
        for match in matches:
            if match[1]:
                inner_list.append({"rna": match[1]})
            if match[4]:
                inner_list.append({"dna": match[4]})
            if match[7]:
                inner_list.append({"cell_line": match[7]})
            if match[10]:
                inner_list.append({"protein": match[10]})
            if match[13]:
                inner_list.append({"cell_type": match[13]})
        l_predictions.append(inner_list)
        
    result = []
    for inner_list in l_targets:
        outcome_inner = []
        for word in inner_list:
            found = False
            for dict_list in l_predictions:
                for dict_item in dict_list:
                    if word.lower() in dict_item.values():
                        outcome_inner.append(list(dict_item.keys())[0])
                        found = True
                        break
                if found:
                    break
            if not found:
                outcome_inner.append("O")
        result.append(outcome_inner)
    return result

    
l = [[('Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number'), ('of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'), ('glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes'), ('and', 'and', 'and', 'and', 'and', 'and', 'and', 'and'), ('their', 'their', 'their', 'their', 'their', 'their', 'their', 'their'), ('sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity'), ('to', 'to', 'to', 'to', 'to', 'to', 'to', 'to'), ('hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone'), ('action', 'action', 'action', 'action', 'action', 'action', 'action', 'action'), ('.', '.', '.', '.', '.', '.', '.', '.')], [('The', 'The', 'The', 'The', 'The', 'The', 'The', 'The'), ('study', 'study', 'study', 'study', 'study', 'study', 'study', 'study'), ('demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated'), ('a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'), ('decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased'), ('level', 'level', 'level', 'level', 'level', 'level', 'level', 'level'), ('of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'), ('glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors'), ('(', '(', '(', '(', '(', '(', '(', '('), ('GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR'), (')', ')', ')', ')', ')', ')', ')', ')'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes'), ('from', 'from', 'from', 'from', 'from', 'from', 'from', 'from'), ('hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic'), ('subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects'), (',', ',', ',', ',', ',', ',', ',', ','), ('and', 'and', 'and', 'and', 'and', 'and', 'and', 'and'), ('an', 'an', 'an', 'an', 'an', 'an', 'an', 'an'), ('elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated'), ('level', 'level', 'level', 'level', 'level', 'level', 'level', 'level'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('patients', 'patients', 'patients', 'patients', 'patients', 'patients', 'patients', 'patients'), ('with', 'with', 'with', 'with', 'with', 'with', 'with', 'with'), ('acute', 'acute', 'acute', 'acute', 'acute', 'acute', 'acute', 'acute'), ('myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial'), ('infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction'), ('.', '.', '.', '.', '.', '.', '.', '.')]]
#i = ['protein: glucocorticoid receptors; cell_type: lymphocytes', 'protein: glucocorticoid receptors; protein: gr; cell_type: peripheral blood lymphocytes']
i = ['Gleichzeitig wurde der totale Inhalt von t lymphocytes um 1,5 % in pe', 'i-hydroxyvitamin d3 ( 1-1.5 mg daily , within 4']
label_pred(i, l) 
#['Gleichzeitig wurde der totale Inhalt von t lymphocytes um 1,5 % in pe', 'i-hydroxyvitamin d3 ( 1-1.5 mg daily , within 4']
#x = ['cell_type: lymphocytes; protein: glucocorticoid receptors; rna: rna1', 'protein: glucocorticoid receptors; dna: dna1; cell_line: cell1']
#x = ['cell_type: lymphocytes; protein: glucocorticoid receptors', 'protein: glucocorticoid receptors; protein: gr; cell_type: peripheral blood lymphocytes']

def label_true(incoming, actual):
    l_targets = [[tuple_list[0] for tuple_list in sublist] for sublist in actual]
    l_predictions = [[{e.split(":")[0].strip(): e.split(":")[1].strip()} for e in x.split(";") if e] for x in incoming]

    outcome = []
    for inner_list in l_targets:
        outcome_inner = []
        for word in inner_list:
            found = False
            for dict_list in l_predictions:
                for dict_item in dict_list:
                    if word.lower() in dict_item.values():
                        outcome_inner.append(list(dict_item.keys())[0])
                        found = True
                        break
                if found:
                    break
            if not found:
                outcome_inner.append("O")
        outcome.append(outcome_inner)
    print(l_predictions)

#l = [[('Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number'), ('of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'), ('glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes', 'lymphocytes'), ('and', 'and', 'and', 'and', 'and', 'and', 'and', 'and'), ('their', 'their', 'their', 'their', 'their', 'their', 'their', 'their'), ('sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity', 'sensitivity'), ('to', 'to', 'to', 'to', 'to', 'to', 'to', 'to'), ('hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone', 'hormone'), ('action', 'action', 'action', 'action', 'action', 'action', 'action', 'action'), ('.', '.', '.', '.', '.', '.', '.', '.')], [('The', 'The', 'The', 'The', 'The', 'The', 'The', 'The'), ('study', 'study', 'study', 'study', 'study', 'study', 'study', 'study'), ('demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated', 'demonstrated'), ('a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'), ('decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased', 'decreased'), ('level', 'level', 'level', 'level', 'level', 'level', 'level', 'level'), ('of', 'of', 'of', 'of', 'of', 'of', 'of', 'of'), ('glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors', 'glucocorticoid receptors'), ('(', '(', '(', '(', '(', '(', '(', '('), ('GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR', 'GR'), (')', ')', ')', ')', ')', ')', ')', ')'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes', 'peripheral blood lymphocytes'), ('from', 'from', 'from', 'from', 'from', 'from', 'from', 'from'), ('hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic', 'hypercholesterolemic'), ('subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects', 'subjects'), (',', ',', ',', ',', ',', ',', ',', ','), ('and', 'and', 'and', 'and', 'and', 'and', 'and', 'and'), ('an', 'an', 'an', 'an', 'an', 'an', 'an', 'an'), ('elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated', 'elevated'), ('level', 'level', 'level', 'level', 'level', 'level', 'level', 'level'), ('in', 'in', 'in', 'in', 'in', 'in', 'in', 'in'), ('patients', 'patients', 'patients', 'patients', 'patients', 'patients', 'patients', 'patients'), ('with', 'with', 'with', 'with', 'with', 'with', 'with', 'with'), ('acute', 'acute', 'acute', 'acute', 'acute', 'acute', 'acute', 'acute'), ('myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial', 'myocardial'), ('infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction', 'infarction'), ('.', '.', '.', '.', '.', '.', '.', '.')]]
#label_true(['protein: glucocorticoid receptors; cell_type: lymphocytes', 'protein: glucocorticoid receptors; protein: gr; cell_type: peripheral blood lymphocytes'], l) 



