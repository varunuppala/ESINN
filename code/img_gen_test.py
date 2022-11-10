import matplotlib.pyplot as plt
import numpy as np


class Shape:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.points = set()

    @staticmethod
    def intersects(shape1, shape2):
        c1 = shape1.__class__
        c2 = shape2.__class__
        # print(c1, c2)
        if issubclass(c1, Rectangle) and issubclass(c2, Rectangle):
            return Rectangle.intersects(shape1, shape2)
        if issubclass(c1, Circle) and issubclass(c2, Circle):
            return Circle.intersects(shape1, shape2)
        return len(shape1.points.intersection(shape2.points)) > 0


class Rectangle(Shape):
    def __init__(self, x, y, w, h) -> None:
        super().__init__(x, y)
        self.w, self.h = w, h
        for i in range(x, x+w):
            for j in range(y, y+h):
                self.points.add((i,j))

    @staticmethod
    def intersects(rect1, rect2):
        # print("\tChecking Rectangle Intersection")
        if rect1.x+rect1.w+1 < rect2.x:
            return False
        if rect1.x > rect2.x+rect2.w+1:
            return False
        if rect1.y+rect1.h+1 < rect2.y:
            return False
        if rect1.y > rect2.y+rect2.h+1:
            return False
        return True


class Square(Rectangle):
    def __init__(self, x, y, s) -> None:
        super().__init__(x, y, s, s)
        self.s = s


class Ellipse(Shape):
    def __init__(self, x, y, rw, rh):
        super().__init__(x, y)
        self.rw, self.rh = rw, rh
        for i in range(x-rw, x+rw):
            for j in range(y-rh, y+rh):
                if ((x-i) / rw)**2 + ((y-j) / rh)**2 < 1:
                    self.points.add((i,j))


class Circle(Ellipse):
    def __init__(self, x, y, r):
        super().__init__(x, y, r, r)
        self.r = r

    @staticmethod
    def intersects(circ1, circ2):
        # print("\tChecking Circle Intersection")
        return (circ1.x - circ2.x)**2 + (circ1.y - circ2.y)**2 < (circ1.r + circ2.r)**2
        


def rand_shape_params(minx, maxx, miny, maxy, minw, maxw, minh, maxh):
    x = np.random.randint(minx, maxx+1)
    y = np.random.randint(miny, maxy+1)
    w = np.random.randint(minw, maxw+1)
    h = np.random.randint(minh, maxh+1)
    return x, y, w, h


'''
def gen_squares():
    pic = [[1]*m for i in range(n)]
    for c in range(count):
        s = random.randint(sizes[0], sizes[1])
        rx = random.randint(0, n-1-s)
        ry = random.randint(0, m-1-s)
        for i in range(rx, rx+s):
            for j in range(ry, ry+s):
                pic[i][j] = 0
    return pic

def gen_circles():
    pic = [[1]*m for i in range(n)]
    for c in range(count):
        s = random.randint(sizes[0], sizes[1]) // 2
        rx = random.randint(s, n-1-s)
        ry = random.randint(s, m-1-s)
        for i in range(rx - s, rx + s + 1):
            for j in range(ry - s, ry + s + 1):
                if (rx - i)**2 + (ry-j)**2 <= s**2:
                    pic[i][j] = 0
    return pic
'''

if __name__ == '__main__':
    img_dim = 256
    minw, maxw = 10, 50
    minh, maxh = 10, 50
    mincount, maxcount = 1, 20

    ticker = 0

    while True:
        pic_type = np.random.randint(1, 4)
        count = np.random.randint(mincount, maxcount+1)
        shapes = []
        
        if ticker % 100 == 0:
            print(ticker, '\t', count, '\t', pic_type)

        c = 0
        if pic_type == 1:
            # All squares
            while True:
                x, y, s, h = rand_shape_params(0, img_dim-1, 0, img_dim-1, minw, maxw, 0, 0)
                if x+s-1 < img_dim and y+s-1 < img_dim:
                    newshape = Square(x, y, s)
                    if all([not Shape.intersects(newshape, shape) for shape in shapes]):
                        shapes.append(newshape)
                        c+=1
                        if c == count:
                            break
        elif pic_type == 2:
            # All circles
            while True:
                x, y, r, h = rand_shape_params(0, img_dim-1, 0, img_dim-1, minw // 2, maxw // 2, 0, 0)
                if x-r > 0 and y-r > 0 and x+r < img_dim and y+r < img_dim:
                    newshape = Circle(x, y, r)
                    if all([not Shape.intersects(newshape, shape) for shape in shapes]):
                        shapes.append(newshape)
                        c+=1
                        if c == count:
                            break
        elif pic_type == 3:
            # Squares and circles
            while True:
                shape_type = np.random.randint(1,3)
                if shape_type == 1:
                    # Circle
                    x, y, r, h = rand_shape_params(0, img_dim-1, 0, img_dim-1, minw // 2, maxw // 2, 0, 0)
                    if x-r > 0 and y-r > 0 and x+r < img_dim and y+r < img_dim:
                        newshape = Circle(x, y, r)
                        if all([not Shape.intersects(newshape, shape) for shape in shapes]):
                            shapes.append(newshape)
                            c+=1
                            if c == count:
                                break
                else:
                    # Square
                    x, y, s, h = rand_shape_params(0, img_dim-1, 0, img_dim-1, minw, maxw, 0, 0)
                    if x+s-1 < img_dim and y+s-1 < img_dim:
                        newshape = Square(x, y, s)
                        if all([not Shape.intersects(newshape, shape) for shape in shapes]):
                            shapes.append(newshape)
                            c+=1
                            if c == count:
                                break
        
        # circ1 = Circle(128, 128, 30)
        # circ2 = Circle(140, 140, 20)
        # shapes = [circ1, circ2]

        pic = [[1]*img_dim for i in range(img_dim)]
        all_points = set()
        for shape in shapes:
            # print('\t', shape.x, '\t', shape.y, '\t', shape.s)
            all_points = all_points.union(shape.points)
        for ptx, pty in all_points:
            pic[pty][ptx] = 0

        # print('Count:', count, 'Pic Type:', pic_type)
        # plt.imshow(pic, cmap='gray')
        # plt.show()
        # break
        ticker += 1

        


        


'''
pic = gen_circles()

plt.imshow(pic, cmap='gray')
plt.show()
'''