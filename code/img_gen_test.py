# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv, os, time


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
        shape1_buffer = shape1.points.copy()
        for (i,j) in shape1.points:
            shape1_buffer.add((i+1, j))
            shape1_buffer.add((i, j+1))
            shape1_buffer.add((i-1, j))
            shape1_buffer.add((i, j-1))
        return len(shape1_buffer.intersection(shape2.points)) > 0


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

    @staticmethod
    def get_params(img_dim, pix_target, minw, maxw, minh, maxh):
        low = max(minw, round(pix_target / maxh))
        high = min(maxw, round(pix_target / minh)) + 1
        w = np.random.randint(low, high)
        h = round(pix_target / w) #.astype(int)
        x = np.random.randint(1, img_dim - w)
        y = np.random.randint(1, img_dim - h)
        return x, y, w, h

    @staticmethod
    def possible_params(pix_target, minw, maxw, minh, maxh):
        low = max(minw, round(pix_target / maxh))
        high = min(maxw, round(pix_target / minh)) + 1
        return minw * minh <= pix_target and maxw * maxh >= pix_target and low < high



class Square(Rectangle):
    def __init__(self, x, y, s, h) -> None:
        super().__init__(x, y, s, s)
        self.s = s

    @staticmethod
    def get_params(img_dim, pix_target, minw, maxw, minh, maxh):
        w = np.round(np.sqrt(pix_target)).astype(int)
        if minw > w or maxw < w:
            return None

        x = np.random.randint(1, img_dim - w)
        y = np.random.randint(1, img_dim - w)
        return x, y, w, w

    @staticmethod
    def possible_params(pix_target, minw, maxw, minh, maxh):
        w = np.round(np.sqrt(pix_target)).astype(int)
        return minw <= w and maxw >= w


class Ellipse(Shape):
    def __init__(self, x, y, rw, rh):
        super().__init__(x, y)
        self.rw, self.rh = rw, rh
        for i in range(x-rw, x+rw):
            for j in range(y-rh, y+rh):
                if ((x-i) / rw)**2 + ((y-j) / rh)**2 < 1:
                    self.points.add((i,j))

    @staticmethod
    def get_params(img_dim, pix_target, minw, maxw, minh, maxh):
        low = max(round(minw / 2), round(2 * pix_target / maxh / np.pi))
        high = min(round(maxw / 2), round(2 * pix_target / minh / np.pi)) + 1
        rw = np.random.randint(low, high)
        rh = round(pix_target / rw / np.pi) #.astype(int)
        x = np.random.randint(rw + 1, img_dim - rw)
        y = np.random.randint(rh + 1, img_dim - rh)
        return x, y, rw, rh

    @staticmethod
    def possible_params(pix_target, minw, maxw, minh, maxh):
        low = max(round(minw / 2), round(2 * pix_target / maxh / np.pi))
        high = min(round(maxw / 2), round(2 * pix_target / minh / np.pi)) + 1
        return np.pi * minw * minh / 4 <= pix_target and np.pi * maxw * maxh / 4 >= pix_target and low < high


class Circle(Ellipse):
    def __init__(self, x, y, r, h):
        super().__init__(x, y, r, r)
        self.r = r

    @staticmethod
    def intersects(circ1, circ2):
        # print("\tChecking Circle Intersection")
        return (circ1.x - circ2.x)**2 + (circ1.y - circ2.y)**2 < (circ1.r + circ2.r)**2 + 5**2

    @staticmethod
    def get_params(img_dim, pix_target, minw, maxw, minh, maxh):
        r = np.round(np.sqrt(pix_target / np.pi)).astype(int)
        x = np.random.randint(r+1, img_dim - r)
        y = np.random.randint(r+1, img_dim - r)
        return x, y, r, r

    @staticmethod
    def possible_params(pix_target, minw, maxw, minh, maxh):
        r = np.round(np.sqrt(pix_target / np.pi)).astype(int)
        return minw <= 2*r and maxw >= 2*r


class Triangle(Shape):
    def __init__(self, x, y, w, h):
        super().__init__(x, y)
        self.w, self.h = w, h
        for i in range(round(x - w / 2), round(x + w / 2)):
            for j in range(y, y+h):
                if j >= 2 * h / w * (i - x) + y and j >= -2 * h / w * (i - x) + y:
                    self.points.add((i,j))

    @staticmethod
    def get_params(img_dim, pix_target, minw, maxw, minh, maxh):
        low = max(minw, round(2 * pix_target / maxh))
        high = min(maxw, round(2 * pix_target / minh)) + 1
        w = np.random.randint(low, high)
        h = round(2 * pix_target / w)
        x = np.random.randint(round(w / 2) + 1, img_dim - round(w / 2))
        y = np.random.randint(1, img_dim - h)
        return x, y, w, h
    
    @staticmethod
    def possible_params(pix_target, minw, maxw, minh, maxh):
        low = max(minw, round(2 * pix_target / maxh))
        high = min(maxw, round(2 * pix_target / minh)) + 1
        return minw * minh / 2 <= pix_target and maxw * maxh / 2 >= pix_target and low < high


        

'''
def rand_shape_params(minx, maxx, miny, maxy, minw, maxw, minh, maxh):
    x = np.random.randint(minx, maxx+1)
    y = np.random.randint(miny, maxy+1)
    w = np.random.randint(minw, maxw+1)
    h = np.random.randint(minh, maxh+1)
    return x, y, w, h
'''

def gen_image(img_size, shapeset, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr):
    pr = np.random.uniform(minpr, maxpr)
    pixuse = img_size**2 * pr
    count = np.random.randint(mincount, maxcount + 1)

    shape_sizes = None
    attempts = 0
    while shape_sizes is None:
        if attempts >= 100:
            return None
        shape_size_partition = [0, 1] + [np.random.random() for i in range(count - 1)]
        shape_size_partition = sorted(shape_size_partition)
        shape_sizes = [shape_size_partition[i+1] - shape_size_partition[i] for i in range(count)]
        shape_sizes = [pixuse * shape_size for shape_size in shape_sizes]
        shape_sizes = sorted(shape_sizes, reverse=True)

        if not all([any([shape_type.possible_params(size, minw, maxw, minh, maxh) for shape_type in shapeset]) for size in shape_sizes]):
            shape_sizes = None
        attempts += 1

    shape_list = []

    for ss in shape_sizes:
        new_shape = None
        attempts = 0
        while new_shape is None:
            if attempts >= 100:
                return None
            shape_type = None
            while shape_type is None:
                candidate_shape_type = np.random.choice(shapeset)
                if candidate_shape_type.possible_params(ss, minw, maxw, minh, maxh):
                    shape_type = candidate_shape_type
            params = shape_type.get_params(img_size, ss, minw, maxw, minh, maxh)
            if params:
                x, y, w, h = params
                candidate_shape = shape_type(x, y, w, h)
                if all([not Shape.intersects(candidate_shape, shp) for shp in shape_list]):
                    new_shape = candidate_shape
            attempts += 1
        shape_list.append(new_shape)

    pic_array = np.full((img_dim, img_dim), bg_color, dtype=np.float32)
    all_points = set()
    for shape in shape_list:
        all_points = all_points.union(shape.points)
    for ptx, pty in all_points:
        pic_array[pty][ptx] = shape_color

    # print('pr:', pr)
    # print('pixuse:', pixuse)
    # print('total pix:', np.sum(1-pic_array))

    img = Image.fromarray(pic_array, mode='I')

    return img, count


def gen_dataset(name, rngseed, params):
    np.random.seed(rngseed)
    start_time = time.time()

    n, img_size, shapeset, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr = params

    path = './%s'%(name)

    if not os.path.exists(path):
        os.makedirs(path)

    with open('%s/%s.txt'%(path, name), 'w') as f:
        lines = ['RNG Seed:\t\t\t%d'%(rngseed),
                 'Image Count:\t\t%d'%(n),
                 'Image Size:\t\t\t%d x %d'%(img_size, img_size),
                 'Shape Types:\t\t%s'%(', '.join([s.__name__ for s in shapeset])),
                 'Shape Color:\t\t%d'%(shape_color),
                 'Background Color:\t%d'%(bg_color),
                 'Min Shape Width:\t%d'%(minw),
                 'Max Shape Width:\t%d'%(maxw),
                 'Min Shape Height:\t%d'%(minh),
                 'Max Shape Height:\t%d'%(maxh),
                 'Min Shape Count:\t%d'%(mincount),
                 'Max Shape Count:\t%d'%(maxcount),
                 'Min Pixel Coverage:\t%.2f'%(minpr),
                 'Max Pixel Coverage:\t%.2f'%(maxpr)]
        for line in lines:
            f.write(line + '\n')

    report_freq = 100
    label_list = [['image_path', 'shape_count']]

    for id in range(n):
        img_name = '%s_%d'%(name, id)
        result = None
        while result is None:
            result = gen_image(img_size, shapeset, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
        img, count = result
        label_list.append(['./%s/%s'%(name, img_name), count])
        img.save('%s/%s.png'%(path, img_name))
        if id % report_freq == 0 and id != 0:
            time_passed = time.time() - start_time
            hours = time_passed // 60**2
            mins = time_passed // 60 % 60
            secs = time_passed % 60
            print('Image %s complete -- %d:%02d:%02d'%(img_name, hours, mins, secs))

    with open('%s/%s_labels.csv'%(path, name), 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(label_list)

    time_passed = time.time() - start_time
    hours = time_passed // 60**2
    mins = time_passed // 60 % 60
    secs = time_passed % 60
    print('Dataset %s complete -- %d:%02d:%02d'%(name, hours, mins, secs))

    return


if __name__ == '__main__':
    n = 10000
    img_dim = 256
    minw, maxw = 10, 60
    minh, maxh = 10, 60
    mincount, maxcount = 1, 20
    minpr, maxpr = 0.02, 0.7

    shape_set = [Circle, Square, Triangle]
    shape_color = 1
    bg_color = 0
    params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
    gen_dataset('CircSqrTri_WonB', 0, params)

    shape_set = [Circle, Square, Triangle, Ellipse, Rectangle]
    params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
    gen_dataset('CircSqrTriRecElp_WonB', 0, params)

    shape_set = [Circle]
    params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
    gen_dataset('Circ_WonB', 0, params)

    shape_set = [Circle, Square, Triangle, Ellipse, Rectangle]
    shape_color = 0
    bg_color = 1
    params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
    gen_dataset('CircSqrTriRecElp_BonW', 0, params)

    shape_set = [Circle]
    params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)
    gen_dataset('Circ_BonW', 0, params)
