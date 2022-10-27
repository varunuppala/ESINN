from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random

fig = plt.figure()

shape = "square"

n=100
patches = []

class Polygon:
	def __init__(self,x,y,s):
		self.x = x
		self.y = y
		self.s = s
	def make_squares(self):
		return patches.append(matplotlib.patches.Rectangle((self.x, self.y),self.s,self.s,),)
	def make_circles(self):
		return patches.append(matplotlib.patches.Circle((self.x, self.y),self.s,),)
	def make_pixels(self):
		return patches.append(matplotlib.patches.Circle((self.x, self.y),self.s,),)

def plot(patches):
	ax = fig.add_subplot(111, aspect='equal')
	
	plt.xlim([0, 1000])
	plt.ylim([0, 1000])

	p = PatchCollection(patches)

	ax.add_collection(PatchCollection(patches))

	plt.axis('off')
	plt.style.use('grayscale')
	plt.show()

def main():
	for i in range(0,n):
	    x = random.uniform(1, 1000)
	    y = random.uniform(1, 1000)
	    s = random.uniform(1,20)
	    p = Polygon(x,y,s)
	    if shape == "circle":
	    	p.make_circles()
	    elif shape == "square":
	    	p.make_squares()
	    else:
	    	p.make_pixels()

	plot(patches)

main()

