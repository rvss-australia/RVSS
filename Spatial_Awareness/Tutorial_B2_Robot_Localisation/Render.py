import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from PIL import Image

# utility function that returns the coordinates for an arrow using Plotly lines
# the Nones cause a break in the line
def arrow(x,y,theta,size):
  c=np.cos(theta)
  s=np.sin(theta)
  return([x,
          x+c*size,
          x+(0.75*c-0.15*s)*size,
          None,
          x+c*size,
          x+(0.75*c+0.15*s)*size,
          None],
         [y,
          y+s*size,
          y+(0.75*s+0.15*c)*size,
          None,
          y+s*size,
          y+(0.75*s-0.15*c)*size,
          None])

#utility function that returns the coordinates for an ellipse
#defined by pos and covariance.  Ellipse has numpts vertices
# which means it has numpts+2 coordinates returned
# (repeat first coord and then a None to break from the next ellipse or whatever)
def ellipse(pos,covariance,numpts=20,scale=1):
  a = covariance[0,0]
  b = covariance[0,1]
  c = covariance[1,1]
  cp = np.sqrt(c)
  bp = b/cp
  ap = np.sqrt(a-bp*bp)
  sqrt=np.array([[ap,bp],[0,cp]])
  xs,ys=[],[]
  for i in range(numpts+1):
    theta = i*2*np.pi/numpts
    circlepos = np.array([scale*np.cos(theta),scale*np.sin(theta)])
    coord = np.matmul(sqrt,circlepos)+pos
    xs.append(coord[0])
    ys.append(coord[1])
  xs.append(None)
  ys.append(None)  
  return (xs,ys)

def draw_robot(state,xcol,ycol,rx,ry,ra,rc):
  arrowsize=0.1
  for t in range(state.shape[0]):
    x=state[t,0]
    y=state[t,1]
    theta=state[t,2]
    arrx,arry = arrow(x, y, theta, arrowsize)
    rx+=arrx
    ry+=arry
    ra+=[t]*len(arrx)
    rc+=[xcol]*len(arrx)
    arrx,arry = arrow(x, y, theta+np.pi/2, arrowsize)
    rx+=arrx
    ry+=arry
    ra+=[t]*len(arrx)
    rc+=[ycol]*len(arrx)
  
def draw_meas(state,m,col,rx,ry,ra,rc):
  for t in range(len(m)):
    rpos = state[t,0:2]
    rtheta = state[t,2]
    c=np.cos(rtheta)
    s=np.sin(rtheta)
    rrot = np.array([[c,-s],[s,c]])

    mms=m[t]
    numpts=20
    for mm in mms:
      mpos = mm.position[:,0]
      #mcov = mm.covariance
      mwpos = rpos+np.matmul(rrot,mpos)
      #mwcov = np.matmul(np.matmul(rrot,mcov),rrot.transpose())
      rx+=[rpos[0],mwpos[0]]
      ry+=[rpos[1],mwpos[1]]
      ra+=[t,t]
      rc+=[col,col]
    rx+=[None]
    ry+=[None]
    ra+=[t]
    rc+=[col]
    
def draw_rcov(state, rcov, col, rx, ry, ra, rc):
  for t in range(len(state)):
    rpos = state[t,0:2]
    cov = rcov[t,0:2,0:2]
    xs, ys = ellipse(rpos, cov)
    rx+=xs
    ry+=ys
    ra+=[t]*len(xs)
    rc+=[col]*len(xs)

def draw_landmarks(landmarks, marker_cov, col, rx, ry, ra, rc):
  for t in range(len(landmarks)):
    lms=landmarks[t]
    lmcs=marker_cov[t]
    for l in range(lms.shape[1]):
      lm=lms[:,l]
      lmc=lmcs[2*l:2*l+2,2*l:2*l+2]
      xs,ys = ellipse(lm,lmc)
      rx+=xs
      ry+=ys
      ra+=[t]*len(xs)
      rc+=[col]*len(xs)

def Render(state=None, gt_state=None, measurements=None, robot_cov=None, landmarks=None, marker_cov=None):
  x,y,a,c=[],[],[],[]

  if state is not None:
    draw_robot(state,'r','g',x,y,a,c)
    
  if gt_state is not None:
    draw_robot(gt_state,'dr','dg',x,y,a,c)

  if measurements is not None:
    draw_meas(state, measurements,'b',x,y,a,c)

  if robot_cov is not None:
    draw_rcov(state,robot_cov,'c',x,y,a,c)

  if landmarks is not None:
    draw_landmarks(landmarks, marker_cov,'b',x,y,a,c)

  colmap={'r':'#FF0000','g':'#00FF00','dr':'#800000','dg':'#008000','b':'#8080ff', 'c':'#00ffff'}
  fig = px.line(x=x,y=y,animation_frame=a, color=c, color_discrete_map=colmap)

  marker_files = [filename for filename in os.listdir('./image') if filename.startswith("M")]
  marker_world_width = 0.3
  for i,filename in enumerate(marker_files):
    fprts = filename.split('_')
    xpos = float(fprts[1])
    ypos = float(fprts[2])
    im = Image.open('image/' + filename)
    imsize = 0.3
    # Add images
    fig.add_layout_image(
        dict(
            source = im,
            xref="x",
            yref="y",
            x=xpos-imsize/2,
            y=ypos+imsize/2,
            sizex=imsize,
            sizey=imsize,
            sizing="contain",
            opacity=1.0,
            layer="below")
    )
  fig.update_layout(sliders=[{"currentvalue": {"prefix": "Timestep="}}])
  fig.update_xaxes(showgrid=False, range = [-4,-2])
  fig.update_yaxes(showgrid=False, range = [-3.5,-1.5])
  fig.update_layout(template="plotly_white", showlegend=False)
  fig.update_layout(width = 500, height=530) #makes it look square on my screen
  fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
  fig.show()
