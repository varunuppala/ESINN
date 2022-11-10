from matplotlib.collections import PatchCollection
import matplotlib
import matplotlib.pyplot as plt
import random

fig = plt.figure()

shape = "square"

n=30
patches = []
img_size = 1000

class Polygon:
	def __init__(self,x,y,s):
		self.x = x
		self.y = y
		self.s = s
	def make_squares(self):
		return patches.append(matplotlib.patches.Rectangle((self.x, self.y),self.s,self.s,facecolor=(1,1,0)),)
	def make_circles(self):
		return patches.append(matplotlib.patches.Circle((self.x, self.y),self.s,),)
	def make_pixels(self):
		return patches.append(matplotlib.patches.Circle((self.x, self.y),self.s,),)

def plot(patches):
	ax = fig.add_subplot(111, aspect='equal')
	
	plt.xlim([0, img_size])
	plt.ylim([0, img_size])

	p = PatchCollection(patches)

	ax.add_collection(PatchCollection(patches, color='k'))

	plt.axis('off')
	# plt.style.use('grayscale')
	plt.savefig('img1.png', dpi=275, bbox_inches='tight', pad_inches=0)
	plt.show()

def main():
	for i in range(0,n):
		x = random.uniform(1, img_size)
		y = random.uniform(1, img_size)
		s = random.uniform(20,50)
		p = Polygon(x,y,s)
		if shape == "circle":
			p.make_circles()
		elif shape == "square":
			p.make_squares()
		else:
			p.make_pixels()
	plot(patches)
	
	
main()

