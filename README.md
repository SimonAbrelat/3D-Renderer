# 3D-Renderer

Notes on Issues with the code:
1. I am using the painter's algorithm so there is minor artifacting around the corners of objects
    I think this is okay, and I'm not sure how much of it is related to my process or the model triangles
    getting flipped when they were made.
2. The lack of ambient lighting makes everything really dark. I think that is just the nature of this basic lighting.
3. Sometimes my axes are confusing and I often have to negate to get translations moving in the correct direction
4. Either my camera is too close to my objects, the FOV is too low, or my perspective is wrong because I often get somewhat
    severe warping.

In the images folder there are images with the following settings:

eiffel_90.png
- Camera at [0,0,-50] looking towards the origin 
- Light at [100,100,0] with full white light
- Model translated [0,-20,0] down 20, and scaled by [.5,.5,.5]
- FOV is 90deg, this is meant to show the perspective nature

eiffel_180.png
- Same variables as other eiffel image
- FOV is 180deg to how how the FOV changes the extreme perspective

mew_warped.png
- Camera at [20,20,-20] looking at [0,30,0])
- Light at [0,100,-50] with color [255,255,0]
- Model at origin scaled by [3,1,5] and rotated [0,0,pi/2]
- FOV is 100 deg

mew_spooky.png
- Camera at [0,0,20] looking at origin
- Light  at [0,-50,-50] with color [255,255,0]
- Model translated by [0,0,8], scaled by [3,3,3], and rotated by [0,0,pi/2]

shark_action.png
- Camera at [0,20,-20] looking at origin
- Light at [0,-10,-50] with full white
- Model translated by [5,0,0] scaled by [1.5,1.5,1.5], and rotated by [0,pi/3,pi/4]
