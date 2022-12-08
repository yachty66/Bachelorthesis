class MyersBriggsType:
    def __init__(self, mind, energy, nature, tactics, identity):
        self.mind = mind
        self.energy = energy
        self.nature = nature
        self.tactics = tactics
        self.identity = identity
        
        
        
# now I should be able to initialize mbti characters


intj_t = MyersBriggsType("Intuition", "Thinking", "Judging", "Introverted", "T")
intj_a = MyersBriggsType("Intuition", "Thinking", "Judging", "Introverted", "A")
intp_t = MyersBriggsType("Intuition", "Thinking", "Perceiving", "Introverted", "T")
intp_a = MyersBriggsType("Intuition", "Thinking", "Perceiving", "Introverted", "A")
entj_t = MyersBriggsType("Extroverted", "Thinking", "Judging", "Intuition", "T")
entj_a = MyersBriggsType("Extroverted", "Thinking", "Judging", "Intuition", "A")
entp_t = MyersBriggsType("Extroverted", "Thinking", "Perceiving", "Intuition", "T")
entp_a = MyersBriggsType("Extroverted", "Thinking", "Perceiving", "Intuition", "A")

#which game can I develop here? simple things like a quiz to determine your mbti type or a game that uses your mbti type to determine your character in a game
#I could give each type based on his personality certain abilities and disadvantages. for example 

print(intj_t.mind)