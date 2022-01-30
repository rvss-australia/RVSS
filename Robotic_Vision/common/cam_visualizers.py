# from common import Camera
import numpy as np
import math
# ---------------------------------------------------------------------------------------#
class CamVisualizer:
    """
    Class for visualizer of a camera object. Used to generate frustrums in Matplotlib
    
        Constructor:
        `CamVisualizer(parameters)
            camera Camera object being visualized
            f_length  length of the frustrum
            fb_width  width of base of frustrum (camera centre end)
            ft_width  width of top of frustrum (lens end)
        Methods:  
          gen_frustrum_poly()  return 4x4x3 matrix of points to create Poly3DCollection with Matplotlib
                               Order of sides created [top, right, bottom, left]
    """
    def __init__(self, camera, f_length=0.1, fb_width=0.05, ft_width=0.1):
        """
        Create instance of CamVisualizer class
        
        Required parameters:
            camera  Camera object being visualized (see common.py for Camera class)
        
        Optional parameters:
            f_length length of the displayed frustrum (0.1 default)
            fb_width width of the base of displayed frustrum (camera centre end) (0.05 default)
            ft_width width of the top of displayed frustrum (lens end) (0.1 default)
        """
        self.camera = camera
        
        # Define corners of polygon in cameras frame (cf) in homogenous coordinates
        # b is base t is top rectangle
        self.cf_b0 = np.array([-fb_width/2,-fb_width/2, 0, 1]).reshape(4,1)
        self.cf_b1 = np.array([-fb_width/2, fb_width/2, 0, 1]).reshape(4,1)
        self.cf_b2 = np.array([fb_width/2, fb_width/2, 0, 1]).reshape(4,1)
        self.cf_b3 = np.array([fb_width/2,-fb_width/2, 0,1 ]).reshape(4,1)
        self.cf_t0 = np.array([-ft_width/2, -ft_width/2, f_length, 1]).reshape(4,1)
        self.cf_t1 = np.array([-ft_width/2, ft_width/2, f_length, 1]).reshape(4,1)
        self.cf_t2 = np.array([ft_width/2, ft_width/2, f_length, 1]).reshape(4,1)
        self.cf_t3 = np.array([ft_width/2, -ft_width/2, f_length, 1]).reshape(4,1)
    
    def gen_frustrum_poly(self):
        
        # Transform frustrum points to world coordinate frame using the camera extrinsics
        
        b0 = (self.camera.pose @ self.cf_b1)[:-1].flatten()
        b1 = (self.camera.pose @ self.cf_b2)[:-1].flatten()
        b2 = (self.camera.pse @ self.cf_b3)[:-1].flatten()
        b3 = (self.camera.pose @ self.cf_b0)[:-1].flatten()
        t0 = (self.camera.pose @ self.cf_t1)[:-1].flatten()
        t1 = (self.camera.pose @ self.cf_t2)[:-1].flatten()
        t2 = (self.camera.pose @ self.cf_t3)[:-1].flatten()
        t3 = (self.camera.pose @ self.cf_t0)[:-1].flatten()
        
        # Each set of four points is a single side of the Frustrum
        points = np.array([[b0,b1,t1,t0], [b1,b2,t2,t1],[b2,b3,t3,t2],[b3,b0,t0,t3]])
        return points

def next_cam_pose(frame_id, camera, num_steps, final_pose, pose_mat=False):
    """
    final_pose - list - [x,y,z,ax,ay,az] (rotx, roty, angles given in degrees) or 4x4 homogeneous matrix (indicated by pose_mat)
    num_steps - int - number of steps left until end of sequence ( > 0)
    pose_mat - boolean - indicator whether we are using a homogeneous matrix (True) or list as above (False) defaults to False
    TODO Neaten EVERYTHING!
    """
    if num_steps == 1:
        if pose_mat:
            camera.T = final_pose
            return
        
        x,y,z,ax,ay,az = final_pose
            
    # Currently assuming all input is valid
    # Probably better way to do this but this is what I have for now
    else:
        # Calculate the angles we are currently in
        # Using decomposition technique found here: http://nghiaho.com/?page_id=846
        # TODO remove toing and froing from degrees and radians
        # TODO make neater approach for final_pose confusions
        
        # Get current camera poses
        cx,cy,cz, cax, cay, caz = decompose_h_mat(camera.T)
        # Get final camera poses in list form
        if pose_mat:
            fx, fy, fz, fax, fay, faz = decompose_h_mat(final_pose)
        else:
            fx, fy, fz, fax, fay, faz = final_pose
            
        # OLD WAY BACKUP
#         current_ax = math.degrees(math.atan2(camera.T[2,1], camera.T[2,2]))
#         current_ay = math.degrees(math.atan2(-1*camera.T[2,0], np.sqrt(camera.T[2,1]**2 + camera.T[2,2]**2)))
#         current_az = math.degrees(math.atan2(camera.T[1,0], camera.T[0,0]))
            
        # Calculate the difference between where we are and where we want to go
        # determine the size of one step given number of steps left and increase stat by step size
        x = cx + ((fx - cx)/num_steps)
        y = cy + ((fy - cy)/num_steps)
        z = cz + ((fz - cz)/num_steps)
        ax = cax + ((fax - cax)/num_steps)
        ay = cay + ((fay - cay)/num_steps)
        az = caz + ((faz - caz)/num_steps)
        
    # Calculate the final h_mat based on the angles and rotations we should be at
    thetax = math.radians(ax)
    thetay = math.radians(ay)
    thetaz = math.radians(az)

    cx, sx = np.cos(thetax), np.sin(thetax)
    cy, sy = np.cos(thetay), np.sin(thetay)
    cz, sz = np.cos(thetaz), np.sin(thetaz)

    rotx = np.array([[1, 0, 0], 
                     [0, cx, -sx],
                     [0, sx, cx]])
    roty = np.array([[cy, 0, sy],
                     [0, 1, 0],
                     [-sy, 0, cy]])
    rotz = np.array([[cz, -sz, 0],
                     [sz, cz, 0],
                     [0, 0, 1]])
    rot = rotz @ roty @ rotx

    trans = np.array([x,y,z]).reshape(3,1)
    camera.T = np.vstack((np.hstack((rot, trans)), np.array([0,0,0,1])))
    
def decompose_h_mat(h_mat):
    """
    Decompose homogeneous matrix into rotation around each axis and translation components.
    Note: Currently assumes final rot = rz @ ry @ rx
    Gives angles in degrees for now (probably needs tidy up)
    """
    ax = math.degrees(math.atan2(h_mat[2,1], h_mat[2,2]))
    ay = math.degrees(math.atan2(-1*h_mat[2,0], np.sqrt(h_mat[2,1]**2 + h_mat[2,2]**2)))
    az = math.degrees(math.atan2(h_mat[1,0], h_mat[0,0]))
    x = h_mat[0, 3]
    y = h_mat[1, 3]
    z = h_mat[2, 3]
    
    return x,y,z,ax,ay,az

def polar2cartesian(radius, azimuth, inclination):
    # input in degrees
    # radius [0, inf.]
    # inclination[0 180]
    # azimuth [0 360]
    
    # Note inclination angle is from the highest point
    x = radius*np.sin(math.radians(inclination))*np.cos(math.radians(azimuth))
    y = radius*np.sin(math.radians(inclination))*np.sin(math.radians(azimuth))
    z = radius*np.cos(math.radians(inclination))
    return np.array([x,y,z])
    
