'''
l_targets
[['However', ',', 'there', 'was', 'a', 'significantly', 'positive', 'association', 'between', 'tumor beta 2-M', 'expression', 'and', 'the', 'degree', 'of', 'lymphocytic', 'infiltration', 'in', 'the', 'tumor', 'tissue', '.'], ['Beta 2-M', 'serum', 'levels', 'were', 'determined', 'by', 'an', 'enzyme-linked', 'immunosorbent', 'assay', 'in', 'samples', 'from', '22', 'of', 'the', 'above', 'women', '.'], ['Although', 'some', 'of', 'the', 'highest', 'values', 'had', 'been', 'obtained', 'in', 'women', 'with', 'larger', '(', 'T4', ')', 'primary', 'tumors', ',', 'the', 'authors', 'failed', 'to', 'detect', 'any', 'statistical', 'relationship', 'between', 'beta 2-M', 'expression', 'in', 'the', 'tumor', 'with', 'serum', 'levels', 'or', 'between', 'serum beta 2-M', 'and', 'the', 'above', 'histologic', ',', 'laboratory', ',', 'and', 'clinical', 'factors', '.'], ['[', 'Preliminary', 'observation', 'of', 'level', 'free-form E receptor', 'levels', 'in', 'serum', 'of', 'normal', 'childbearing-aged', 'and', 'pregnant', 'women', ']'], ['In', '137', 'cases', 'of', 'childbearing-aged', 'and', 'pregnant', 'women', ',', 'free form E receptor', 'levels', '(', 'sE', ')', 'in', 'serum', 'were', 'measured', 'by', 'ELISA', '.'], ['The', 'level', 'of', 'sE', 'was', 'significantly', 'decreased', 'during', 'the', 'first', 'trimester', ',', 'slightly', 'higher', 'in', 'the', 'second', 'trimester', ',', 'and', 'recovered', 'to', 'normal', 'in', 'the', 'third', 'trimester', '.'], ['The', 'level', 'remained', 'lower', 'in', '29', 'PIH', 'women', 'but', 'appeared', 'higher', 'in', 'overdue', 'pregnancies', 'as', 'compared', 'with', 'the', 'normal', '3rd', 'trimester', 'range', '.'], ['The', 'results', 'indicate', 'that', 'there', 'is', 'a', 'relationship', 'between', 'a', 'change', 'in', 'T cell', 'function', 'and', 'pregnancy', '.']]

l_predictions
[[{'protein': 'glucocorticoid receptors'}, {'cell_type': 'lymphocytes'}], [{'protein': 'glucocorticoid receptors'}, {'protein': 'gr'}, {'cell_type': 'peripheral blood lymphocytes'}], [{'cell_type': 'lymphocytes'}, {'protein': 'gr'}, {'cell_type': 'control cells'}], [{'protein': 'gr'}], [{'cell_type': 'lymphocytes'}, {'protein': 'gr'}], [{'protein': '1 , 25-dihydroxyvitamin d3 receptors'}, {'cell_type': 'lymphocytes'}, {'cell_type': 't- and b-lymphocyte'}], [{'cell_type': 'lymphocytes'}], [{'cell_type': 't lymphocytes'}]]
'''
import re
def label_true():
    
    #l_targets = [
    #    [tuple_list[0] for tuple_list in sublist] for sublist in actual
    #]
    #
    l_targets = [['However', ',', 'there', 'was', 'a', 'significantly', 'positive', 'association', 'between', 'tumor beta 2-M', 'expression', 'and', 'the', 'degree', 'of', 'lymphocytic', 'infiltration', 'in', 'the', 'tumor', 'tissue', '.'], ['Beta 2-M', 'serum', 'levels', 'were', 'determined', 'by', 'an', 'enzyme-linked', 'immunosorbent', 'assay', 'in', 'samples', 'from', '22', 'of', 'the', 'above', 'women', '.'], ['Although', 'some', 'of', 'the', 'highest', 'values', 'had', 'been', 'obtained', 'in', 'women', 'with', 'larger', '(', 'T4', ')', 'primary', 'tumors', ',', 'the', 'authors', 'failed', 'to', 'detect', 'any', 'statistical', 'relationship', 'between', 'beta 2-M', 'expression', 'in', 'the', 'tumor', 'with', 'serum', 'levels', 'or', 'between', 'serum beta 2-M', 'and', 'the', 'above', 'histologic', ',', 'laboratory', ',', 'and', 'clinical', 'factors', '.'], ['[', 'Preliminary', 'observation', 'of', 'level', 'free-form E receptor', 'levels', 'in', 'serum', 'of', 'normal', 'childbearing-aged', 'and', 'pregnant', 'women', ']'], ['In', '137', 'cases', 'of', 'childbearing-aged', 'and', 'pregnant', 'women', ',', 'free form E receptor', 'levels', '(', 'sE', ')', 'in', 'serum', 'were', 'measured', 'by', 'ELISA', '.'], ['The', 'level', 'of', 'sE', 'was', 'significantly', 'decreased', 'during', 'the', 'first', 'trimester', ',', 'slightly', 'higher', 'in', 'the', 'second', 'trimester', ',', 'and', 'recovered', 'to', 'normal', 'in', 'the', 'third', 'trimester', '.'], ['The', 'level', 'remained', 'lower', 'in', '29', 'PIH', 'women', 'but', 'appeared', 'higher', 'in', 'overdue', 'pregnancies', 'as', 'compared', 'with', 'the', 'normal', '3rd', 'trimester', 'range', '.'], ['The', 'results', 'indicate', 'that', 'there', 'is', 'a', 'relationship', 'between', 'a', 'change', 'in', 'T cell', 'function', 'and', 'pregnancy', '.']]
    #for x in incoming:
    #    result = re.split(";(?![^\(]*\))", x)
    #    result = [x.strip() for x in result]
    #    l_predictions.append([{e.split(":")[0].strip(): e.split(":")[1].strip()} for e in result if e])
    l_predictions =  [[{'protein': 'glucocorticoid receptors'}, {'cell_type': 'lymphocytes'}], [{'protein': 'glucocorticoid receptors'}, {'protein': 'gr'}, {'cell_type': 'peripheral blood lymphocytes'}], [{'cell_type': 'lymphocytes'}, {'protein': 'gr'}, {'cell_type': 'control cells'}], [{'protein': 'gr'}], [{'cell_type': 'lymphocytes'}, {'protein': 'gr'}], [{'protein': '1 , 25-dihydroxyvitamin d3 receptors'}, {'cell_type': 'lymphocytes'}, {'cell_type': 't- and b-lymphocyte'}], [{'cell_type': 'lymphocytes'}], [{'cell_type': 't lymphocytes'}]]
    result = []
    for inner_list in l_targets:
        #print(inner_list)
        outcome_inner = []
        for word in inner_list:
            #print(word)
            found = False
            for dict_list in l_predictions:
                #print(dict_list)
                for dict_item in dict_list:
                    #print(dict_item)
                    if word.lower() in dict_item.values():
                        print(word)
                        outcome_inner.append(list(dict_item.keys())[0])
                        found = True
                        break
                if found:
                    break
            if not found:
                outcome_inner.append("O")
        result.append(outcome_inner)
    return result
        
        
        
label_true()