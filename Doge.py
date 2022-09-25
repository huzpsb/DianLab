class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_score(self, period):
        print('%s在第%s学期的总分是%s.' % (self.name, period, self.score))

    def wander(self):
        if self.score < 10:
            self.score = 0
        else:
            self.score -= 10

    def study(self):
        if self.score < 90:
            self.score += 10
        else:
            self.score = 100


class Gzh(Student):
    def print_score(self, period):
        print('%s在第%s学期的总分是%s.' % (self.name, period, 100))


xiaoMing = Student("小明", 80)
xiaoMing.study()
xiaoMing.study()
xiaoMing.study()
xiaoMing.study()
xiaoMing.print_score("1")
xiaoMing.wander()
xiaoMing.print_score("2")
xiaoMing.wander()
xiaoMing.wander()
xiaoMing.study()
xiaoMing.print_score("3")
me = Gzh("gzh", 80)
me.wander()
me.wander()
me.print_score("1")
