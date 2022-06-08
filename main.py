from tkinter import W
from PIL import Image, ImageDraw
import numpy as np
import math

#------------------------------------------------------------------
# CONSTANTS 
#------------------------------------------------------------------
aspect = 16.0/9.0
#aspect = 1.0
fov = (2 * np.pi) / 9 # 40 degs
imgy = 1080 
imgx = int(imgy * aspect)

#------------------------------------------------------------------
# Utility Matrix Functions 
#------------------------------------------------------------------
def trans(x,y,z):
    return np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [x,y,z,1]
    ])

def scale(x,y,z):
    return np.array([
        [x,0,0,0],
        [0,y,0,0],
        [0,0,z,0],
        [0,0,0,1]
    ])

def rotate(x,y,z):
    return np.array([
        [ np.cos(x),np.sin(x),0,0],
        [-np.sin(x),np.cos(x),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]) @ np.array([
        [np.cos(y),0,-np.sin(y),0],
        [0,1,0,0],
        [np.sin(y),0, np.cos(y),0],
        [0,0,0,1]
    ]) @ np.array([
        [1,0,0,0],
        [0, np.cos(z),np.sin(z),0],
        [0,-np.sin(z),np.cos(z),0],
        [0,0,0,1]
    ])

def to3d(vec4): return vec4[0:3]
def to2d(vec4): return vec4[0:2]

def toPix(vec4): return tuple(int(v) for v in vec4[0:2])

def unit(v): return v / np.linalg.norm(v)
def length(v): return np.linalg.norm(v)

#------------------------------------------------------------------
# CLASSES
#------------------------------------------------------------------
class Camera():
    perspective = np.ones((4,4))
    orientation = np.ones((4,4))

    def __init__(self, aspect, fov, near, far, width, height):
        self.fov = fov 
        self.aspect = aspect
        self.near = near
        self.far = far
        self.width = width
        self.height =  height

        assert width == int(height * aspect)

        trig_y = np.arctan(fov/2.0)
        trig_x = float(aspect) * trig_y
        scale = float(far)/float(far-near)

        '''
        scale = 1.0/float(far-near)
        self.perspective = np.array([
            [2.0/float(width), 0,                 0,                  0],
            [0,                2.0/float(height), 0,                  0],
            [0,                0,                 scale,              0],
            [0,                0,                 -float(near)*scale, 1]
        ])
        '''

        self.perspective = np.array([
            [trig_x, 0,      0,                  0],
            [0,      trig_y, 0,                  0],
            [0,      0,      scale,              1],
            [0,      0,      -scale*float(near), 0]
        ])

        w_2 = float(width) / 2.0
        h_2 = float(height) / 2.0
        d_2 = float(far-near) / 2.0
        self.window = np.array([
            [w_2, 0,   0,   0],
            [0,   h_2, 0,   0],
            [0,   0,   d_2, 0],
            [w_2, h_2, d_2, 1]
        ])

    def look_at(self, eye, at):
        self.eye = eye
        zaxis = unit(at - eye)
        xaxis = unit(np.cross(np.array([0,1,0]), zaxis))
        yaxis = np.cross(zaxis, xaxis)
        
        zaxis = -zaxis
        self.orientation = np.array([
            [xaxis[0]           , yaxis[0]           , zaxis[0]           , 0],
            [xaxis[1]           , yaxis[1]           , zaxis[1]           , 0],
            [xaxis[2]           , yaxis[2]           , zaxis[2]           , 0],
            [-np.dot(xaxis, eye), -np.dot(yaxis, eye), -np.dot(zaxis, eye), 1],
        ])

    def toClip(self, model):
        model.transform(self.orientation)

    def toNDC(self, model):
        model.transform(self.perspective)
        model.scale(3)

    def toWindow(self, model):
        model.transform(self.window)

    def view(self, model):
        model.transform(self.orientation)
        model.transform(self.perspective)
        model.apply(2)
        model.transform(self.window)

class Triangle():

    def __init__(self, v0, v1, v2):
        s1 = v1[0:3] - v0[0:3]
        s2 = v2[0:3] - v0[0:3]
        if (length(np.cross(s1,s2)) == 0):
            self.vertices = [np.array([0,0,0,1]),np.array([0,0,0,1]),np.array([0,0,0,1])]
            self.color = (0,0,0)
            self.normal = np.array([0,0,0])
            return

        self.normal = unit(np.cross(s1,s2))
        self.vertices = [v0,v1,v2]
        self.color = (0,0,0)

    
    def light(self, pos, color, m_diff):
        l_v = unit(to3d(self.vertices[0]) - pos)
        dot = np.dot(self.normal, l_v)
        self.color = tuple([int(x) for x in (color * m_diff * max(0, dot))])

    def pixels(self):
        return [tuple(int(x) for x in v[0:2]) for v in self.vertices]

    def mult(self, mat):
        self.vertices[:] = [v @ mat for v in self.vertices]

    def div(self, idx):
        self.vertices[:] = [v / v[idx] for v in self.vertices]

    def __repr__(self):
        return f"Triangle:\n\tVerts:{self.vertices}\n\tNorm:{self.normal}\n\tColor:{self.color}\n"
    
class Model():
    triangles = []
    normals = []
    colors = []

    def __init__(self, diffuse):
        self.m_diff = diffuse

    def parse(self, file):
        with open(file, "r") as f:
            for l in f.readlines():
                # String Operations
                nums = [float(x) for x in l.split() if x not in ['', '\n']]
                assert len(nums) == 9
                vertices = [np.append(nums[i:i+3], [1]) for i in range(0, len(nums), 3)]
                self.triangles.append(Triangle(*vertices))
                '''
                # Convert list into triangles (sets of verts)
                vertices = [np.array(nums[i:i+3]) for i in range(0, len(nums), 3)]
                self.triangles.append([np.append(v, [1]) for v in vertices])

                # Compute normals for later operations                
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                self.normals.append(unit(np.cross(v1,v2)))
                '''
    
    def modify(self, trans, scale, rotate):
        self.transform(trans @ scale @ rotate)
    
    def transform(self, mat):
        for tri in self.triangles: tri.mult(mat)

    def scale(self, idx):
        for tri in self.triangles: tri.div(idx)

    def color(self, light):
        for tri in self.triangles:
            tri.light(light.position, light.color, self.m_diff)
    
    def cull(self, eye):
        self.triangles = list(tri for tri in self.triangles if np.dot(tri.normal, eye) > 0)
    
    def draw(self):
        self.triangles.sort(key=lambda x:x.vertices[0][2])
        #return [(tri.color) for tri in self.triangles]
        return [(tri.pixels(), tri.color) for tri in self.triangles]

class Light():
    def __init__(self, pos, col):
        self.position = pos
        self.color = col



'''
tri = Triangle(np.array([-.2,0,.2,1]), np.array([.2,0,.2,1]),np.array([0,.4,0,1]))
for v in tri.vertices:
    print(v)
print(tri.normal)
'''
light = Light(np.array([100,100,0]), np.array([255,255,255]))

cam = Camera(aspect, fov, 1, 100, imgx, imgy)
cam.look_at(np.array([0,0,-50]), np.array([0,0,0]))

mod = Model(1)
mod.parse("assets/eiffel_ag.raw")

# World Space Transformations
# mod.transform(rotate(0,0,0))
# mod.transform(scale(3,3,3))
# mod.transform(trans(-20,0,0))
mod.modify(trans(0,0,0), scale(1,1,1), rotate(0,0,0))

# Backface Culling
# mod.cull(cam.eye)

# View Transformation
cam.toClip(mod)

# Lighting Transformation
mod.color(light)

# Screen Transformation
cam.toNDC(mod)
cam.toWindow(mod)

#------------------------------------------------------------------
# Image Generation 
#------------------------------------------------------------------
image = Image.new("RGB", (imgx, imgy))
draw = ImageDraw.Draw(image)

for (pix, col) in mod.draw():
    draw.polygon(pix, fill = col)

image.save("test.png", "PNG")