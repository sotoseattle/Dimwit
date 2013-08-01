
class Rvar(object):
  
    def __init__(self, variable, card, opts = [None]):
        self.id = variable
        if card.__class__ == int:
          self.cd = range(0,card)
        elif card.__class__ == list:
          self.cd = card
        else:
          raise Exception("wrong cardinality")
    
    def totCard(self):
        return len(self.cd)
    
    def __repr__(self):
        return "%s" % self.id
        
    def __lt__(self, other):
        return self.id < other.id

    __str__ = __repr__

