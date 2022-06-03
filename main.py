from PIL import Image, ImageDraw
import numpy as np

#------------------------------------------------------------------
# CONSTANTS 
#------------------------------------------------------------------
aspect = 16/9
fov = (2 * np.pi) / 9 # 40 degs
imgx = 720 
imgy = int(imgx * aspect)

#------------------------------------------------------------------
# Utility Matrix Functions 
#------------------------------------------------------------------
def trans(x,y,z):
    return np.matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [x,y,z,1]
    ])

def scale(x,y,z):
    return np.matrix([
        [x,0,0,0],
        [0,y,0,0],
        [0,0,z,0],
        [0,0,0,1]
    ])

def rotate(x,y,z):
    return np.matrix([
        [ np.cos(x),np.sin(x),0,0],
        [-np.sin(x),np.cos(x),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]) * np.matrix([
        [np.cos(x),0,-np.sin(x),0],
        [0,1,0,0],
        [np.sin(x),0, np.cos(x),0],
        [0,0,0,1]
    ]) * np.matrix([
        [1,0,0,0],
        [0, np.cos(x),np.sin(x),0],
        [0,-np.sin(x),np.cos(x),0],
        [0,0,0,1]
    ])

def perspective(a,r,n,f):
    trig = np.arctan(a/2)
    scale = f/(f-n)
    return np.matrix([
        [trig/r,0,0,0],
        [0,trig,0,0],
        [0,0,scale,1],
        [0,0,-scale*n,0]
    ])

def to3d(vec4):
    return np.array(vec4).ravel()[0:3]

def to2d(vec4):
    return np.array(vec4).ravel()[0:2]

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

        assert height == int(width * aspect)

        self.perspective = perspective(aspect,fov,near,far)
        self.window = np.array([
            [width/2,       0,             0, 0],
            [0,      height/2,             0, 0],
            [0,      0,       (far - near)/2, 0],
            [width/2,height/2,(far + near)/2, 1]
        ])

    def set_orientation(self, p, v):
        tmp = np.array([0,1,0])
        f = v - p
        self.eye = f
        f = f / np.linalg.norm(f)
        r = np.cross(tmp, f)
        r = r / np.linalg.norm(r)
        u = np.cross(f, r)
        u = u / np.linalg.norm(u)
        #print("r:", r, "len:", np.linalg.norm(r))
        #print("u:", u, "len:", np.linalg.norm(u))

        self.orientation = np.linalg.inv(np.array([
            [r[0],r[1],r[2],0],
            [u[0],u[1],u[2],0],
            [f[0],f[1],f[2],0],
            [p[0],p[1],p[2],1]
        ]))

    def toClip(self, model):
        model.transform(self.orientation)

    def toNDC(self, model):
        model.transform(self.perspective)
        model.apply(lambda x: x / float(x.T[3]))

    def toWindow(self, model):
        model.transform(self.window)

    def view(self, model):
        model.transform(self.orientation @ self.perspective)
        model.apply(lambda x: x / float(x.T[3]))
        model.transform(self.window)

class Model():
    triangles = []
    def __init__(self, diffuse):
        self.m_diff = diffuse

    def parse(self, file):
        with open(file, "r") as f:
            for l in f.readlines():
                tri = Triangle()
                tri.parse(l)
                if tri is not None:
                    self.triangles.append(tri)
    
    def transform(self, mat):
        for tri in self.triangles: 
            tri.transform(mat)

    def apply(self, fn):
        for tri in self.triangles:
            tri.apply(fn)
    
    def append(self, tri):
        self.triangles.append(tri)

class Triangle():
    def __init__(self): pass

    def __repr__(self):
        rep = ""
        for i in range(len(self.vertices)):
            rep += "v" + str(i+1) + ": ["
            for j in range(len(self.vertices[i])):
                rep += str(self.vertices[i][j])
                if (j < 3): rep += ", "
            rep += "]"
            if (i < 2): rep += ", "
        return rep

    def parse(self, s):
        nums = [float(x) for x in s.split() if x not in ['', '\n']]
        assert len(nums) == 9
        self.vertices = [np.array(nums[i:i+3] + [1]) for i in range(0, len(nums), 3)]

        v1 = to3d(self.vertices[1]) - to3d(self.vertices[0])
        v2 = to3d(self.vertices[2]) - to3d(self.vertices[0])
        self.normal = np.cross(v1, v2)

    def transform(self, mat):
        self.vertices[:] = (v @ mat for v in self.vertices)

    def apply(self, fn):
        self.vertices[:] = (fn(v) for v in self.vertices)

    def sub(self, vec3):
        return to3d(self.vertices[0]) - vec3
    
    def draw(self):
        return [tuple(to2d(v)) for v in self.vertices]

    def light(self, light, diffuse):
        dot = np.dot(self.normal, self.sub(light.position))
        return tuple([int(x) for x in (light.color * diffuse * max(0, dot))])

class Light():
    def __init__(self, pos, color):
        self.position = pos
        self.color = color

light = Light(np.array([0,0,10]), np.array([255,0,0]))

mod = Model(1/(4*np.pi))
mod.parse("assets/shark_ag.raw")
#mod.transform(rotate(0,np.pi,0))
mod.transform(scale(.5,.5,.5))
#mod.transform(trans(0,-5,20))

cam = Camera(aspect, fov, 1, 100, imgx, imgy)
cam.set_orientation(np.array([0,0,10]), np.array([0,0,0]))
cam.view(mod)

'''
So right now the axes are kind of confusing
it seems like the the camera orientation v1 is the side, v2 in front, and v3 above
'''

#------------------------------------------------------------------
# Image Generation 
#------------------------------------------------------------------
image = Image.new("RGB", (imgy, imgx))
draw = ImageDraw.Draw(image)

for tri in mod.triangles:
    if (np.dot(tri.normal, tri.sub(cam.eye)) > 0):
        draw.polygon(tri.draw(), fill = tri.light(light,mod.m_diff))

image.save("test.png", "PNG")