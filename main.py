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
    ]) * np.array([
        [np.cos(x),0,-np.sin(x),0],
        [0,1,0,0],
        [np.sin(x),0, np.cos(x),0],
        [0,0,0,1]
    ]) * np.array([
        [1,0,0,0],
        [0, np.cos(x),np.sin(x),0],
        [0,-np.sin(x),np.cos(x),0],
        [0,0,0,1]
    ])

def perspective(a,r,n,f):
    trig = np.arctan(a/2)
    scale = f/(f-n)
    return np.array([
        [trig/r,0,0,0],
        [0,trig,0,0],
        [0,0,scale,1],
        [0,0,-scale*n,0]
    ])

def to3d(vec4): return vec4[0:3]
def to2d(vec4): return vec4[0:2]

def unit(v): return v / np.linalg.norm(v)

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
        f = unit(v - p)
        r = unit(np.cross(np.array([0,1,0]), f))
        u = unit(np.cross(f, r))

        self.eye = f
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
        model.transform(self.orientation)
        model.transform(self.perspective)
        model.scale(3)
        model.transform(self.window)

class Model():
    triangles = []
    normals = []

    def __init__(self, diffuse):
        self.m_diff = diffuse

    def parse(self, file):
        with open(file, "r") as f:
            for l in f.readlines():
                # String Operations
                nums = [float(x) for x in l.split() if x not in ['', '\n']]
                assert len(nums) == 9

                # Convert list into triangles (sets of verts)
                vertices = [np.array(nums[i:i+3]) for i in range(0, len(nums), 3)]
                self.triangles.append([np.append(v, [1]) for v in vertices])

                # Compute normals for later operations                
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                self.normals.append(unit(np.cross(v1,v2)))
    
    def transform(self, mat):
        self.triangles[:] = ([v @ mat for v in verts] for verts in self.triangles)

    def scale(self, idx):
        self.triangles[:] = ([v / v[idx] for v in verts] for verts in self.triangles)

    def draw(self, light):
        ret = []
        for i in range(len(self.triangles)):
            # Compute light Vector
            l_v = unit(light.position - to3d(self.triangles[i][0]))
            # Compute Color
            dot = np.dot(self.normals[i], l_v)
            color = tuple([int(x) for x in (light.color * self.m_diff * max(0, dot))])
            # Flatten Vectors to 2d
            pixels = [tuple(to2d(v)) for v in self.triangles[i]]
            ret.append((pixels, color))
        return ret
    
    def cull(self, eye):
        self.triangles = list(self.triangles[i] for i in range(len(self.triangles)) if np.dot(self.normals[i], eye) > 0)

class Light():
    def __init__(self, pos, color):
        self.position = pos
        self.color = color

light = Light(np.array([100,100,0]), np.array([255,255,255]))

cam = Camera(aspect, fov, 1, 100, imgx, imgy)
cam.set_orientation(np.array([0,0,10]), np.array([0,0,0]))

mod = Model(1)
mod.parse("assets/mew_lp.raw")
#mod.transform(rotate(0,np.pi,0))
#mod.transform(scale(2,2,2))
#mod.scale(2)
#mod.transform(trans(0,-5,20))
#mod.cull(cam.eye)
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

for (pix, color) in mod.draw(light):
    draw.polygon(pix, fill = color)
'''
    if (np.dot(tri.normal, tri.sub(cam.eye)) > 0):
        draw.polygon(tri.draw(), fill = tri.light(light,mod.m_diff))
'''

image.save("test_new.png", "PNG")