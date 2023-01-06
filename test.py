mapper = {
        "O": 0,
        "B-DNA": 1,
        "I-DNA": 2,
        "B-RNA": 3,
        "I-RNA": 4,
        "B-cell_line": 5,
        "I-cell_line": 6,
        "B-cell_type": 7,
        "I-cell_type": 8,
        "B-protein": 9,
        "I-protein": 10
    }

#set value of each key to the key but lower case
mapper = {k.lower(): k for k in mapper.keys()}

print(mapper)



