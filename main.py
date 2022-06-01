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
        '''
        for tri in range(len(model.triangles)):
            for p in range(len(model.triangles[tri])):
                model.triangles[tri][p] = model.triangles[tri][p] @ self.perspective 
                model.triangles[tri][p] = model.triangles[tri][p] / float(model.triangles[tri][p].T[3]) 
        '''
        model.transform(self.perspective)
        model.apply(lambda x: x / float(x.T[3]))


    def toWindow(self, model):
        model.transform(self.window)

    #def toDraw(self, model):
    #    return [[tuple(vert[0:2]) for vert in tri] for tri in model.triangles]


class Model():
    triangles = [[]]

    def __init__(self): pass

    def parse(self, file):
        with open(file, "r") as f:
            for (c,l) in enumerate(f.readlines()):
                if (c != 0): self.triangles.append([])
                nums = [float(x) for x in l.split() if x not in ['', '\n']]
                self.triangles[c] = [np.array(nums[i:i+3] + [1]) for i in range(0, len(nums), 3)]
    
    def transform(self, mat):
        for tri in range(len(self.triangles)):
            for p in range(len(self.triangles[tri])):
                self.triangles[tri][p] = self.triangles[tri][p] @ mat

    def apply(self, fn):
        for tri in range(len(self.triangles)):
            for p in range(len(self.triangles[tri])):
                self.triangles[tri][p] = fn(self.triangles[tri][p])
    
    def append(self, tri):
        if len(self.triangles) == 1:
            self.triangles[0] = tri
        else:
            self.triangles.append(tri)
    
    def toDraw(self):
        '''
        acc = []
        for tri in range(len(self.triangles)):
            acc.append([])
            for p in self.triangles[tri]:
                print(
                acc[tri].append(p[0])
        '''
        return [[tuple(p.tolist()[0][0:2]) for p in tri] for tri in self.triangles]

mew = Model()
mew.parse("assets/Mewtwo_lp.raw")
#mew.append(np.array([[-.5,-.5,-.5,1],[0,1,.5,1],[.5,.5,.5,1]]))

#mew.transform(rotate(np.pi /4,0,np.pi/3))
#mew.transform(scale(10,10,10))
mew.transform(trans(1,1,1))

cam = Camera(aspect, fov, 1, 100, imgx, imgy)
cam.set_orientation(np.array([20,50,20]), np.array([0,2,0]))
cam.toClip(mew)
cam.toNDC(mew)
cam.toWindow(mew)
#mew.toList()
#print(mew.toList())
#print(mew.triangles)
#print([list(tri) for tri in mew.triangles])

#------------------------------------------------------------------
# Image Generation 
#------------------------------------------------------------------
image = Image.new("RGB", (imgy, imgx))
draw = ImageDraw.Draw(image)

triangles = mew.toDraw()
print(triangles)
for tri in triangles:
    draw.polygon(tri, fill = (255,255,255))

image.save("test.png", "PNG")